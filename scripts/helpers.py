import pandas as pd
import numpy as np

def read_data(generate_speed_angle=False):
    df = pd.read_parquet("../data/processed/outlier_removed.parquet")
    weather_cols = [col for col in df.columns if col.startswith(("UGRD", "VGRD"))]

    df = df.set_index("forecast_dt")[["rt_plant_id", "production", *weather_cols]]

    if generate_speed_angle:
        for box in ["SW", "NW", "NE", "SE"]:
            df[f"speed_{box}"] = np.sqrt(np.square(df[f"UGRD_80.m.above.ground.{box}"]) + np.square(df[f"VGRD_80.m.above.ground.{box}"]))
            df[f"angle_{box}"] = np.arctan(df[f"UGRD_80.m.above.ground.{box}"] / df[f"VGRD_80.m.above.ground.{box}"])
    return df

def expand_plant_dimension(df):
    PLANTS = sorted(df.rt_plant_id.unique())
    n_loc = len(PLANTS)
    n_time = df.index.nunique()
    cols = [col for col in df.columns if col != "rt_plant_id"]
    n_cols = len(cols)

    df_np = np.zeros((n_time, n_loc, n_cols))
    for i, plant_id in enumerate(PLANTS):
        df_np[:, i, :] = df[df.rt_plant_id == plant_id][cols].values
    return df_np

def split_data(df, train_ratio=0.8, valid_ratio=0.1, scaler=None):    
    time_indices = sorted(df.index.unique())
    
    train_indices = time_indices[:int(len(time_indices) * train_ratio)]
    valid_indices = time_indices[int(len(time_indices) * train_ratio):int(len(time_indices) * (train_ratio + valid_ratio))]
    test_indices = time_indices[int(len(time_indices) * (train_ratio + valid_ratio)):]

    train_df = df.loc[train_indices, :]
    valid_df = df.loc[valid_indices, :]
    test_df = df.loc[test_indices, :]

    train_df_np = expand_plant_dimension(train_df)
    valid_df_np = expand_plant_dimension(valid_df)
    test_df_np = expand_plant_dimension(test_df)
    
    if scaler is not None:
        import pickle
        assert scaler in ["minmax", "standart"]
        if scaler == "minmax":
            from sklearn.preprocessing import MinMaxScaler as scaler_
        else:
            from sklearn.preprocessing import StandartScaler as scaler_
        scalers = {}
        for i, plant in enumerate(PLANTS):
            scalers[plant] = scaler_()
            # train_df = pd.DataFrame(scaler.fit_transform(train_df), index=train_df.index, columns=train_df.columns)
            train_df_np[:, i, :] = scalers[plant].fit_transform(train_df_np[:, i, :])
            valid_df_np[:, i, :] = scalers[plant].transform(valid_df_np[:, i, :])
            test_df_np[:, i, :] = scalers[plant].transform(test_df_np[:, i, :])

        train_df_np = np.array(train_df_np, dtype=np.float32)
        valid_df_np = np.array(valid_df_np, dtype=np.float32)
        test_df_np = np.array(test_df_np, dtype=np.float32)
        
        with open('scalers.pickle', 'wb') as handle:
            pickle.dump(a, handle)

        # with open('scalers.pickle', 'rb') as handle:
        #     b = pickle.load(handle)

    return train_df_np, valid_df_np, test_df_np


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
                if plant is not None:
                    prediction_values = predictions[n, :, plant, label_col_index]
                else:
                    prediction_values = tf.math.reduce_mean(predictions[n, :, :, label_col_index], axis=1)
                
                prediction_values = tf.clip_by_value(prediction_values, clip_value_min=0, clip_value_max=1)

                plt.scatter(self.label_indices, prediction_values, marker='X', edgecolors='k', label='Predictions',c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            # sequence_stride=24, shuffle=False,
            sequence_stride=1, shuffle=True,
            batch_size=64,)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def valid(self):
        return self.make_dataset(self.valid_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result
    


def compile_and_fit(model, window, patience=10, max_epochs=50):
    def wmape(y_true, y_pred):
        total_abs_diff = tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred)))
        total = tf.reduce_sum(y_true)
        wmape = tf.realdiv(total_abs_diff, total)
        return wmape
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min', 
        verbose=1,
        restore_best_weights=True)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError(), wmape]) 
    
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.valid,
                        verbose=1,
                        callbacks=[early_stopping])
    return history