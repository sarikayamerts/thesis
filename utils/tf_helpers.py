import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import subprocess; FOLDER_PATH = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')

gpu_devices = tf.config.list_physical_devices("GPU")

if gpu_devices:
    if tf.config.experimental.get_device_details(gpu_devices[0])["compute_capability"][0] > 6:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # tf.keras.mixed_precision.set_global_policy("float32")
tf.keras.utils.set_random_seed(235813)


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, valid_df, test_df,
                 columns=None, label_columns=None):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.ndim = self.train_df.ndim
        assert self.ndim in [2, 3]

        if columns is None:
            columns = train_df.columns

        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(columns)}
        self.feature_column_indices = [v for k,v in self.column_indices.items() if k not in self.label_columns]
        # self.feature_number = len(self.feature_column_indices)
        if self.train_df.ndim == 2:
            self.number_of_plants = 1
        else:
            self.number_of_plants = self.train_df.shape[1]
        # list(self.label_columns_indices.values())

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.input_shape = (self.input_width, self.number_of_plants, len(self.feature_column_indices))

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :, 1:]
        labels = features[:, self.labels_slice, :, :]
        labels = tf.stack([labels[:, :, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        inputs.set_shape([None, self.input_width, None, None])
        labels.set_shape([None, self.label_width, None, None])
        return inputs, labels

    def plot(self, model=None, plot_col='production', plant=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            if plant is not None:
                input_values = inputs[n, :, plant, plot_col_index]
                label_values = labels[n, :, plant, label_col_index]
            else:
                input_values = tf.math.reduce_mean(inputs[n, :, :, plot_col_index], axis=1)
                label_values = tf.math.reduce_mean(labels[n, :, :, label_col_index], axis=1)

            plt.plot(self.input_indices, input_values, label='Inputs', marker='.', zorder=-10)

            plt.scatter(self.label_indices, label_values, edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                if predictions.ndim == 3:
                    predictions = tf.expand_dims(predictions, axis=2)
                if plant is not None:
                    prediction_values = predictions[n, :, plant, label_col_index]
                else:
                    prediction_values = tf.math.reduce_mean(predictions[n, :, :, label_col_index], axis=1)

                prediction_values = tf.clip_by_value(prediction_values, clip_value_min=0, clip_value_max=1)

                plt.scatter(self.label_indices, prediction_values, marker='X', edgecolors='k', label='Predictions',c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data, test=False):
        data = np.array(data, dtype=np.float32)
        if test:
            sequence_stride, shuffle = 24, False
        else:
            sequence_stride, shuffle = 1, True
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=sequence_stride,
            shuffle=shuffle,
            batch_size=64,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def valid(self):
        return self.make_dataset(self.valid_df, test=True)

    @property
    def test(self):
        return self.make_dataset(self.test_df, test=True)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs[:, -1:, :, 0] + delta


def compile_and_fit(model, window, patience=10, max_epochs=50,
                    loss="mse", optimizer="adam", verbose=1):
    def wmape(y_true, y_pred):
        total_abs_diff = tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(y_true)
        wmape = tf.realdiv(total_abs_diff, total)
        return wmape

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=verbose,
        restore_best_weights=True)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{FOLDER_PATH}/artifacts/checkpoint',
        save_weights_only=True, monitor='val_wmape',
        mode='min',
        verbose=verbose,
        save_best_only=True)

    # metrics = ["mse", wmape]
    model.compile(loss=loss, optimizer=optimizer, metrics=[wmape],
                  steps_per_execution=32, jit_compile=True)

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.valid,
                        verbose=verbose,
                        callbacks=[early_stopping, model_checkpoint])
    model.load_weights(f'{FOLDER_PATH}/artifacts/checkpoint')
    return model, history

def correlation_ordering(df, initial_start=7):
    corr = pd.pivot_table(
        df[["rt_plant_id", "production"]].reset_index(),
        index="forecast_dt", columns="rt_plant_id",
        values="production").corr()
    selected_plant_ids, selected_plants = [], []
    to_append = initial_start
    selected_plant_ids.append(PLANTS[to_append])
    selected_plants.append(to_append)

    for _ in range(93):
        corr_series = corr.iloc[to_append].drop(labels=selected_plant_ids)
        to_append = PLANTS.index(corr_series.idxmax())
        selected_plant_ids.append(PLANTS[to_append])
        selected_plants.append(to_append)
    return selected_plants

# corr_mean = 0.
# out = []

# for i in range(94):
#     corr_list = []
#     selected_plants = []
#     selected_plant_ids = []
#     to_append = i
#     selected_plant_ids.append(PLANTS[to_append])
#     selected_plants.append(to_append)

#     for _ in range(93):
#         corr_series = corr.iloc[to_append].drop(labels=selected_plant_ids)
#         to_append = PLANTS.index(corr_series.idxmax())
#         selected_plant_ids.append(PLANTS[to_append])
#         selected_plants.append(to_append)
#         corr_list.append(np.max(corr_series))

#     corr_mean = np.mean(corr_list)
#     out.append([
#         i, corr_mean, np.min(corr_list), np.sum([i > 0.9 for i in corr_list]),
#         np.sum([i > 0.8 for i in corr_list]), np.sum([i > 0.7 for i in corr_list]),
#         np.sum([i > 0.6 for i in corr_list]), np.sum([i > 0.5 for i in corr_list])])

# out = pd.DataFrame(out)
# out.columns = ["plant", "mean", "min", "above_9", "above_8", "above_7", "above_6", "above_5"]
# out.sort_values("min", ascending=False).head(20)

# selected_plants = correlation_ordering(df)

# def expand_plant_dimension(df, selected_plants=None):
#     n_loc = len(PLANTS)
#     n_time = df.index.nunique()
#     cols = [col for col in df.columns if col != "rt_plant_id"]
#     n_cols = len(cols)

#     df_np = np.zeros((n_time, n_loc, n_cols))
#     if selected_plants is not None:
#         for i, j in enumerate(selected_plants):
#             df_np[:, i, :] = df[df.rt_plant_id == PLANTS[j]][cols].values
#     else:
#         for i, plant_id in enumerate(PLANTS):
#             df_np[:, i, :] = df[df.rt_plant_id == plant_id][cols].values
#     return df_np

# train_df_np = expand_plant_dimension(train_df, selected_plants)
# valid_df_np = expand_plant_dimension(valid_df, selected_plants)
# test_df_np = expand_plant_dimension(test_df, selected_plants)

def _calculate_wmape(pred, actual):
    return np.sum(np.abs(actual - pred)) / np.sum(actual)

def calculate_plantwise_wmape(model, window, selected_plants):
    predictions = model.predict(window.test)
    actuals = np.concatenate([y for _, y in window.test], axis=0)
    predictions_val = model.predict(window.valid)
    actuals_val = np.concatenate([y for _, y in window.valid], axis=0)
    wmape_dict, wmape_dict_val = {}, {}
    for i, j in enumerate(selected_plants):
        try:
            pred_ = predictions[:, :, i, 0].reshape(-1)
            pred_val_ = predictions_val[:, :, i, 0].reshape(-1)
        except:
            pred_ = predictions[:, :, i].reshape(-1)
            pred_val_ = predictions_val[:, :, i].reshape(-1)
        actual_ = actuals[:, :, i, 0].reshape(-1)
        wmape_dict[j] = _calculate_wmape(pred_, actual_)
        actual_val_ = actuals_val[:, :, i, 0].reshape(-1)
        wmape_dict_val[j] = _calculate_wmape(pred_val_, actual_val_)
    wmape_df = pd.DataFrame(wmape_dict.items())
    wmape_val_df = pd.DataFrame(wmape_dict_val.items())
    wmape_df.columns = ["rt_plant_id", "wmape"]
    wmape_val_df.columns = ["rt_plant_id", "wmape_val"]
    wmape_df = pd.merge(wmape_df, wmape_val_df, on="rt_plant_id")
    return wmape_df.sort_values("wmape")

def plot_plantwise_predictions(model, dataset, plant_id=None):
    predictions = model.predict(dataset)
    actuals = np.concatenate([y for _, y in dataset], axis=0)
    if plant_id is not None:
        actual_values = actuals[:, :, plant_id, :].flatten()
        predicted_values = predictions[:, :, plant_id, :].flatten()
    else:
        actual_values = np.mean(actuals, axis=2).flatten()
        predicted_values = np.mean(predictions, axis=2).flatten()
    plt.plot(actual_values[:500], label='Actual')
    plt.plot(predicted_values[:500], label='Predicted')
    plt.legend()

