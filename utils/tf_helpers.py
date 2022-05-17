from abc import abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import wandb
import os

if not os.environ.get("FOLDER_PATH"):
    import subprocess; FOLDER_PATH = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
else:
    FOLDER_PATH = os.environ["FOLDER_PATH"]

gpu = tf.config.list_physical_devices("GPU")

if gpu:
    if tf.config.experimental.get_device_details(gpu[0])["compute_capability"][0] > 6:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # tf.keras.mixed_precision.set_global_policy("float32")
tf.keras.utils.set_random_seed(235813)


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, data):
        self.train_df = data.train_df
        self.valid_df = data.valid_df
        self.test_df = data.test_df
        self.ndim = self.train_df.ndim
        assert self.ndim in [2, 3]
        columns = [col for col in data.df.columns if col != "rt_plant_id"]
        self.data = data

        self.label_columns = ["production"]
        self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(columns)}
        self.feature_column_indices = [v for k,v in self.column_indices.items() if k not in self.label_columns]
        if self.train_df.ndim == 2:
            self.number_of_plants = 1
        else:
            self.number_of_plants = self.train_df.shape[1]

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.input_shape = (self.input_width, self.number_of_plants, len(self.feature_column_indices))
        if shift >= 0:
            self.total_window_size = input_width + shift
        else:
            self.total_window_size = input_width
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        if shift >= 0:
            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
        else:
            self.label_start = self.total_window_size - self.label_width + shift
            self.labels_slice = slice(self.label_start, self.label_start+self.label_width)
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

class BaseTFModel:
    def __init__(self, window):
        self.window = window
        self.OUT_STEPS = 24
        self.history = None
        self.fitted_model = None
        self.valid_predictions = None
        self.test_predictions = None

    @property
    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.window.input_shape)))
        model.add(tf.keras.layers.Permute((1,2,3), name="start------"))

        model = self.add_model(model)

        model.add(tf.keras.layers.Reshape([self.window.number_of_plants, -1], name="end--------"))
        model.add(tf.keras.layers.Dense(self.OUT_STEPS))
        model.add(tf.keras.layers.Permute((2,1)))
        model.add(tf.keras.layers.Reshape([self.OUT_STEPS, self.window.number_of_plants, 1]))
        return model

    @abstractmethod
    def add_model(self, model):
        return model

    def __repr__(self):
        return self.__class__.__name__

    def start_wandb(self, project="keras", name=None, config=None):
        name = name or self.__class__.__name__ + "_" + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=project, name=name, config=config)

    def compile_and_fit(self, patience=10, epochs=50,
                        loss="mse", optimizer="adam", verbose=1):
        model = self.model
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

        if wandb.run is not None:
            wandb_callback = wandb.keras.WandbCallback(
                log_weights=False, # log_gradients=True, # training_data=window.train,
                # validation_data=window.valid, # log_evaluation=True,
            )
            callbacks=[early_stopping, model_checkpoint, wandb_callback]
        else:
            callbacks=[early_stopping, model_checkpoint]

        if gpu:
            model.compile(loss=loss, optimizer=optimizer, metrics=[wmape],
                               steps_per_execution=32, jit_compile=True)
        else:
            model.compile(loss=loss, optimizer=optimizer, metrics=[wmape])

        history = model.fit(self.window.train, epochs=epochs,
                            validation_data=self.window.valid,
                            verbose=verbose, callbacks=callbacks)
        model.load_weights(f'{FOLDER_PATH}/artifacts/checkpoint')
        self.fitted_model = model

    def predict(self):
        self.valid_predictions = self.make_prediction("valid")
        self.test_predictions = self.make_prediction("test")

        if wandb.run is not None:
            wandb.log({
                "valid_predictions": wandb.Table(dataframe=self.valid_predictions),
                "test_predictions": wandb.Table(dataframe=self.test_predictions),
            })

    def make_prediction(self, set="valid"):
        dataset = self.window.__getattribute__(set)
        indices = self.window.data.__getattribute__(f"{set}_indices")
        predictions = self.fitted_model.predict(dataset)
        actuals = np.concatenate([y for _, y in dataset], axis=0)
        xs = indices[-len(actuals[:, :, 0, :].flatten()):]

        prediction_output = pd.DataFrame()

        for i, plant in enumerate(self.window.data.plants):
            pred_ = pd.DataFrame({
                "actuals": np.round(actuals[:, :, i, :].flatten(), 6),
                "predictions": np.round(predictions[:, :, i, :].flatten(), 6),
                "forecast_dt": xs, "rt_plant_id": plant
            })
            prediction_output = prediction_output.append(pred_)
        return prediction_output

    def plot(self, set="valid"):
        preds = self.__getattribute__(f"{set}_predictions")
        if preds is None:
            self.predict()

        fig = go.Figure()

        for plant in self.window.data.plants:
            df_ = self.test_predictions[self.test_predictions["rt_plant_id"] == plant]
            fig.add_trace(go.Scatter(**{"x": df_.forecast_dt, "y": df_.actuals, "name": f"{plant}_actual", "line": {"color": "royalblue"}}))
            fig.add_trace(go.Scatter(**{"x": df_.forecast_dt, "y": df_.predictions, "name": f"{plant}_prediction", "line": {"color": "firebrick"}}))

        fig.update_layout(
            title='Predictions vs. Actuals',
            xaxis_title='Date',
            yaxis_title='Production (MinMaxScaled)')

        if wandb.run is not None:
            wandb.log({f"{set}_prediction_plot": fig})
        else:
            fig.show()

    def _calculate_wmape(self, pred, actual):
        return np.sum(np.abs(actual - pred)) / np.sum(actual)

    def calculate_accuracy(self, set="valid"):
        preds = self.__getattribute__(f"{set}_predictions")
        preds["month"] = pd.to_datetime(preds["forecast_dt"]).dt.strftime("%Y%m")

        wmape_month_df = preds.groupby(["month", "rt_plant_id"], as_index=False).apply(
            lambda x: pd.Series({"wmape": self._calculate_wmape(x["predictions"], x["actuals"])}))

        wmape_df = preds.groupby(["rt_plant_id"], as_index=False).apply(
            lambda x: pd.Series({"wmape": self._calculate_wmape(x["predictions"], x["actuals"])}))
        return wmape_month_df, wmape_df

    def plot_accuracy(self, set=None, wmape_month_df=None, wmape_df=None):
        if (wmape_month_df is None) or (wmape_df is None):
            wmape_month_df, wmape_df = self.calculate_accuracy(set)

        colors = px.colors.qualitative.Plotly
        xs = sorted(wmape_month_df.month.unique())

        fig1 = go.Figure()
        for i, plant in enumerate(self.window.data.plants):
            fig1.add_trace(go.Scatter(**{
                "x": xs,
                "y": wmape_month_df[wmape_month_df["rt_plant_id"] == plant].wmape,
                "name": f"{plant}_wmape", "line": {"color": colors[i%len(colors)]}}))
        fig1.update_layout(xaxis_title='Month', yaxis_title='WMAPE', title="Plantwise WMAPE for months")
        # fig1.update_yaxes(range=[0.1, 0.9])

        fig2 = go.Figure()
        for month_ in xs:
            fig2.add_trace(go.Box(
                y=wmape_month_df[wmape_month_df["month"] == month_].wmape,
                name=month_))
        fig2.update_layout(xaxis_title='Month', yaxis_title='WMAPE', title="Total WMAPE for months")
        # fig2.update_yaxes(range=[0.1, 0.9])

        fig3 = go.Figure()
        fig3.add_trace(go.Box(y=wmape_df.wmape, name=f"{set}_wmape"))
        fig3.update_layout(yaxis_title='WMAPE', title="WMAPE")
        # fig3.update_yaxes(range=[0.1, 0.9])

        if wandb.run is not None:
            wandb.log({f"{set}_wmape_table": wandb.Table(dataframe=wmape_df),
                       f"{set}_wmape_month_table": wandb.Table(dataframe=wmape_month_df)})
            wandb.log({"scatter": fig1, "month_box": fig2, "box": fig3})
        else:
            fig1.show(); fig2.show(); fig3.show();




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
    wmape_df = wmape_df.sort_values("wmape")
    if wandb.run is not None:
        wandb.log({"wmape_table": wandb.Table(dataframe=wmape_df)})
    return wmape_df

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

