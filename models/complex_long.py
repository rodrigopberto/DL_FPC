import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


class CompConv1D(tf.keras.layers.Layer):
    def __init__(self,filters=1,kernel_size=1,strides=1,padding='valid',dilation_rate=1,
        activation=None,use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',
        trainable=True,**kwargs):
        super(CompConv1D, self).__init__()
        self.convreal = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,
            padding=padding,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,
            kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,trainable=trainable)
        self.convimag = tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,
            padding=padding,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,
            kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,trainable=trainable)

    def call(self, input_tensor):
        ureal, uimag = tf.split(input_tensor, num_or_size_splits=2, axis=2)
        oreal = self.convreal(ureal) - self.convimag(uimag)
        oimag = self.convimag(ureal) + self.convreal(uimag)
        x = tf.concat([oreal, oimag], axis=2)
        return x

    def get_config(self):
        config = {
            "convreal": self.convreal,
            "convimag": self.convimag
        }
        base_config = super(CompConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ComplexModel:

    def __init__(self):
        pass


    def load_data(self,x_data=None,x_data_phase=None,y_data=None):

        x_data = np.einsum("klij->kijl",x_data).flatten().reshape(-1,1024)
        x_data_phase = np.einsum("klij->kijl",x_data_phase).flatten().reshape(-1,1024)
        x_dev = x_data[:int(x_data.shape[0]*0.8)]
        x_dev_phase = x_data_phase[:int(x_data_phase.shape[0]*0.8)]
        x_train = x_dev[:int(x_dev.shape[0]*0.8)]
        x_train_phase = x_dev_phase[:int(x_dev_phase.shape[0]*0.8)]
        x_val = x_dev[int(x_dev.shape[0]*0.8):]
        x_val_phase = x_dev_phase[int(x_dev_phase.shape[0]*0.8):]
        x_test = x_data[int(x_data.shape[0]*0.8):]
        x_test_phase = x_data_phase[int(x_data_phase.shape[0]*0.8):]

        ## data for y
        y_data_freq = y_data[:,0,:,:].flatten()
        y_data_phase = y_data[:,1,:,:].flatten()
        y_dev_freq = y_data_freq[:int(y_data_freq.shape[0]*0.8)]
        y_dev_phase = y_data_phase[:int(y_data_phase.shape[0]*0.8)]
        y_train_freq = y_dev_freq[:int(y_dev_freq.shape[0]*0.8)]
        y_train_phase = y_dev_phase[:int(y_dev_phase.shape[0]*0.8)]
        y_val_freq = y_dev_freq[int(y_dev_freq.shape[0]*0.8):]
        y_val_phase = y_dev_phase[int(y_dev_phase.shape[0]*0.8):]
        y_test_freq = y_data_freq[int(y_data_freq.shape[0]*0.8):]
        y_test_phase = y_data_phase[int(y_data_phase.shape[0]*0.8):]


        ##normalize data
        mean_train = np.absolute(x_train).mean()
        std_train = np.absolute(x_train).std()
        mean_train_phase = np.absolute(x_train_phase).mean()
        std_train_phase = np.absolute(x_train_phase).std()


        x_train = (x_train-mean_train)/std_train
        x_val = (x_val-mean_train)/std_train
        x_test = (x_test-mean_train)/std_train

        x_train_phase = (x_train_phase-mean_train_phase)/std_train_phase
        x_val_phase = (x_val_phase-mean_train_phase)/std_train_phase
        x_test_phase = (x_test_phase-mean_train_phase)/std_train_phase

        ## expand dims for cnn
        x_train = np.expand_dims(x_train,-1)
        x_val = np.expand_dims(x_val,-1)
        x_test = np.expand_dims(x_test,-1)
        x_train_phase = np.expand_dims(x_train_phase,-1)
        x_val_phase = np.expand_dims(x_val_phase,-1)
        x_test_phase = np.expand_dims(x_test_phase,-1)

        #real and imaginary division
        x_train = np.concatenate([np.real(x_train),np.imag(x_train)],axis=2)
        x_val = np.concatenate([np.real(x_val),np.imag(x_val)],axis=2)
        x_test = np.concatenate([np.real(x_test),np.imag(x_test)],axis=2)
        x_train_phase = np.concatenate([np.real(x_train_phase),np.imag(x_train_phase)],axis=2)
        x_val_phase = np.concatenate([np.real(x_val_phase),np.imag(x_val_phase)],axis=2)
        x_test_phase = np.concatenate([np.real(x_test_phase),np.imag(x_test_phase)],axis=2)


        #save to class
        self.x_train = x_train
        self.x_val=x_val
        self.x_test=x_test
        self.x_train_phase=x_train_phase
        self.x_val_phase=x_val_phase
        self.x_test_phase=x_test_phase
        self.y_train_freq=y_train_freq
        self.y_val_freq=y_val_freq
        self.y_test_freq=y_test_freq
        self.y_train_phase=y_train_phase
        self.y_val_phase=y_val_phase
        self.y_test_phase=y_test_phase



    def print_model_summary(self):
        m = self.complex_cnn()
        print(m.summary())
        del m


    def complex_cnn(self):
        input_layer = tf.keras.layers.Input(shape=(1024,2))
        conv11 = CompConv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu')(input_layer)
        conv12 = CompConv1D(filters=8, kernel_size=5, strides=1, padding='same', activation='relu')(conv11)
        mp_1 = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(conv12)

        conv21 = CompConv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu')(mp_1)
        conv22 = CompConv1D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu')(conv21)
        mp_2 = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(conv22)

        conv31 = CompConv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(mp_2)
        conv32 = CompConv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(conv31)
        mp_3 = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(conv32)

        conv41 = CompConv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(mp_3)
        conv42 = CompConv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(conv41)
        mp_4 = tf.keras.layers.MaxPool1D(pool_size=2, padding='valid')(conv42)


        flatten = tf.keras.layers.Flatten()(mp_4)

        d_1 = tf.keras.layers.Dense(1024, kernel_initializer='normal', activation='relu')(flatten)
        d_2 = tf.keras.layers.Dense(512, kernel_initializer='normal', activation='relu')(d_1)
        out = tf.keras.layers.Dense(1, kernel_initializer='normal', activation='linear')(d_2)

        model = tf.keras.models.Model(inputs=input_layer, outputs=out)
        return model



    def train(self,batch_size=64,epoch_count=100,scheduler_epoch=20):
        freq_model_name="freq"
        self.freq_model = self.complex_cnn()
        #print(freq_model.summary())
        freq_monitor = tf.keras.callbacks.ModelCheckpoint(freq_model_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
        def scheduler(epoch, lr):
            if epoch%scheduler_epoch == 0 and epoch!=0:
                lr = lr/2
            return lr

        freq_lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

        self.freq_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

        self.freq_model.fit(self.x_train, self.y_train_freq, batch_size=batch_size, epochs=epoch_count, verbose=1, validation_data=(self.x_val, self.y_val_freq), callbacks=[freq_monitor, freq_lr_schedule], shuffle=True)

        phase_model_name="phase"
        self.phase_model = self.complex_cnn()
        #print(freq_model.summary())
        phase_monitor = tf.keras.callbacks.ModelCheckpoint(phase_model_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
        def scheduler(epoch, lr):
            if epoch%scheduler_epoch == 0 and epoch!=0:
                lr = lr/2
            return lr

        phase_lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

        self.phase_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

        self.phase_model.fit(self.x_train_phase, self.y_train_phase, batch_size=batch_size, epochs=epoch_count, verbose=1, validation_data=(self.x_val_phase, self.y_val_phase), callbacks=[phase_monitor, phase_lr_schedule], shuffle=True)

    def get_test_results(self):

        y_predict_freq = self.freq_model.predict(self.x_test)
        y_predict_phase = self.phase_model.predict(self.x_test_phase)

        plot_freq_y_test = self.y_test_freq.flatten()
        plot_phase_y_test = self.y_test_phase.flatten()
        plot_freq_y_pred = y_predict_freq.flatten()
        plot_phase_y_pred = y_predict_phase.flatten()
        freq_error_plot = plot_freq_y_pred-plot_freq_y_test
        phase_error_plot = plot_phase_y_pred-plot_phase_y_test

        df = pd.DataFrame({
            "true_freq":plot_freq_y_test,
            "pred_freq":plot_freq_y_pred,
            "freq_error":freq_error_plot,
            "true_phase":plot_phase_y_test,
            "pred_phase":plot_phase_y_pred,
            "phase_error":phase_error_plot,
        })

        return df



    def save_model_weights(self,freq_model_filename,phase_model_filename):
        self.freq_model.save_weights(freq_model_filename)
        self.phase_model.save_weights(phase_model_filename)
    
    def load_model_from_weights(self,freq_model_filename,phase_model_filename):
        self.freq_model = self.complex_cnn() #tf.keras.models.load_model(freq_model_filename)
        self.freq_model.load_weights(freq_model_filename)
        self.phase_model = self.complex_cnn()
        self.phase_model.load_weights(phase_model_filename) # = tf.keras.models.load_model(phase_model_filename)

        #custom_objects = {"CompConv2D": CompConv2D}
        #with tf.keras.utils.custom_object_scope(custom_objects):
        #    new_model = tf.keras.Model.load_model(freq_model_filename)



        #self.freq_model = tf.keras.models.load_model(freq_model_filename)
        #self.phase_model = tf.keras.models.load_model(phase_model_filename)
    
    def predict_freq(self,x):
        
        x_data = np.einsum("klij->kijl",x).flatten().reshape(-1,1024)

        mean = np.absolute(x_data).mean()
        std = np.absolute(x_data).std()

        x_data = (x_data-mean)/std

        x_data = np.expand_dims(x_data,-1)

        x_data = np.concatenate([np.real(x_data),np.imag(x_data)],axis=2)

        return self.freq_model.predict(x_data,verbose=0).flatten()
    
    def predict_phase(self,x):
        
        x_data = np.einsum("klij->kijl",x).flatten().reshape(-1,1024)

        mean = np.absolute(x_data).mean()
        std = np.absolute(x_data).std()

        x_data = (x_data-mean)/std

        x_data = np.expand_dims(x_data,-1)

        x_data = np.concatenate([np.real(x_data),np.imag(x_data)],axis=2)

        return self.phase_model.predict(x_data,verbose=0).flatten()
