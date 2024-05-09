# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def fragmentar(x, sr, seg):
    import numpy as np
    
    #seg = 20   # fragmentación en cortes de t segundos
    if (len(x)%(seg*sr)) != 0:
        bloques = int(len(x)/((seg*sr))+1)
        muestras = bloques*seg*sr
        #print(minutos)
        n = muestras - len(x)
        #print(muestras, len(x), n)
        zeros = np.zeros(n)
        #print(len(zeros))
        x = np.append(x, zeros)
        #print(len(x))
        
    n = 0
    minuto = seg*sr
    cortes = []
    while n < len(x):
        vector = x[n:n+minuto]
        cortes.append(vector)
        n = n+minuto
    vectores = np.array(cortes)
    
    return vectores, bloques
        

def extract_features(signal, sr, hop_length, frame_length):
    import librosa
    return[
        librosa.feature.rms(signal, hop_length=hop_length, frame_length=frame_length, center=True)[0],
        librosa.feature.spectral_centroid(signal, sr=sr, hop_length=hop_length)[0],
        librosa.feature.zero_crossing_rate(signal, frame_length=frame_length, hop_length=hop_length, center=True)[0],
        librosa.feature.spectral_bandwidth(signal, sr=sr, hop_length=hop_length)[0],
        librosa.feature.spectral_flatness(signal, hop_length=hop_length)[0],
        librosa.feature.spectral_rolloff(signal, sr=sr, hop_length=hop_length)[0]   
    ] 


def vector_features(vectores, sr, hop_length, frame_length):
    features = []
    for v in vectores:
        caracteristicas = []
        rmse, centroide, zeros, band, flat, rolloff = extract_features(v, sr, hop_length, frame_length)
        caracteristicas = [rmse, centroide, zeros, band, flat, rolloff]
        features.append(caracteristicas)
    
    return features


def tiempo(features, sr, hop_length):
    import librosa
    frames = range(len(features[0][0]))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    return t

'''
def ver_features(vectores=0, features=0, bloques=0, minimo=0, maximo=5, t, xlima=0, xlimb=10, ylima=-1, ylimb=1):
    import matplotlib.pyplot as plt
    import librosa
    plt.figure(figsize=(15, 150))
    for i, (v, f) in enumerate(zip(vectores[minimo:maximo], features[minimo:maximo])):
        plt.subplot(bloques, 1, i+1)
        librosa.display.waveplot(v/v.max(), alpha = 0.4)
        plt.plot(t, f[0]/f[0].max(), color='r')
        plt.plot(t, f[1]/f[1].max(), color='y')
        plt.plot(t, f[2]/f[2].max(), color='c')
        plt.plot(t, f[3]/f[3].max(), color='g')
        plt.plot(t, f[4]/f[4].max(), color='m')
        plt.plot(t, f[5]/f[5].max(), color='b')
        plt.xlim(xlima,xlimb)
        plt.ylim(ylima, ylimb)
        plt.grid(axis='x')
        plt.legend(('RMSE', 'centroide', 'zeros', 'bandwidth', 'flatness', 'rolloff'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Corte {}'.format(i))
'''

def cortes(rmses, flats, limite_rmse = 0.02, limite_flat = 0.01):
    import numpy as np
    vector_cortes = []
    for rmse, flat in zip(rmses, flats):
    
        rmse_lim = rmse/rmse.max()
        flatness_lim = flat/flat.max()
        limite_rmse = 0.02
        limite_flat = 0.01
        vector = np.array([])
        for i, (r, f) in enumerate(zip(rmse_lim, flatness_lim)):
            if r < limite_rmse:
                if f > limite_flat:
                    # Añadir fourier
                    vector = np.append(vector, 0)
                else:
                    vector = np.append(vector, 1)
            else:
                vector = np.append(vector, 1)
            
        vector_cortes.append(vector)
        
    return vector_cortes


def recompone_señal_t(vectores, vector_cortes):
    y = np.array([])
    for v in vectores:
        y = np.append(y, v)
    cortes = np.array([])
    for c in vector_cortes:
        cortes = np.append(cortes, c)
        
    frames = range(len(cortes))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    return y, cortes, t

def marca_audio(rmse, flatness):
    import numpy as np
    import stanford_mir; stanford_mir.init()
    
    rmse_lim = rmse/rmse.max()
    flatness_lim = flatness/flatness.max()
    limite_rmse = 0.02
    limite_flat = 0.01
    vector = np.array([])
    for i, (r, f) in enumerate(zip(rmse_lim, flatness_lim)):
        if r < limite_rmse:
            if f > limite_flat:
                vector = np.append(vector, 0)
            else:
                vector = np.append(vector, 1)
        else:
            vector = np.append(vector, 1)
            
    return(vector)
  

def corta_audio(marcas, x, sr, hop_length):
    import numpy as np
    
    cortes = []
    for v in marcas:
        n = 0
        while n < hop_length:
            cortes.append(v)
            n +=1
            
    ceros = np.zeros(abs(len(cortes) - len(x)))
    if len(cortes) > len(x):
        x = np.append(x, ceros)
    if len(cortes) < len(x):
        cortes = np.append(cortes, ceros)
        
    y = cortes * x
    
    cortes3 = []
    cortes_final = []
    for n in y:
        if n != 0:
            cortes3.append(n)
        else:
            cortes_final.append(cortes3)
            cortes3 = []
            
    cortes_array = []
    for n in cortes_final:
        if len(n) > sr/2:
            cortes_array.append(np.array(n))
            
    return(cortes_array)



def cortes_audios(x, sr, hop_length, frame_length):
    import numpy as np
    import librosa, librosa.display
    import stanford_mir; stanford_mir.init()
    
    rmse, centroide, zeros, bandwidth, flatness, rolloff = extract_features(x, sr, hop_length, frame_length)
    
    frames = range(len(rmse))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    marcas = marca_audio(rmse, flatness)
    cortes = corta_audio(marcas, x, sr, hop_length)
    
    return(cortes, frames, t, rmse, centroide, zeros, bandwidth, flatness, rolloff)
    

def grafica(senal, cortes, rmses, flats, t, ax=0, bx=20, ay=-1, by=1, sr=22050):
  
    import matplotlib.pyplot as plt
    import librosa 
    
    plt.figure(figsize=(15, 4))
    
    librosa.display.waveplot(senal/senal.max(), sr=sr)
    plt.plot(t, cortes/cortes.max(), color='k')
    plt.plot(t, rmses/rmses.max(), color='r')
    plt.plot(t, flats/flats.max(), color='g')
    plt.xlim(ax,bx)
    plt.ylim(ay, by)
    plt.xscale('linear')
    plt.grid(axis='x')
    plt.legend(('rmse', 'flatness'), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def fourier(senal, sr = 22050, xlima = 0, xlimb = 22050):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy  
    
    Xag1 = scipy.fft(senal)
    Xag1_mag = np.absolute(Xag1)
    fag1 = np.linspace(0, sr, len(Xag1_mag))
    
    plt.figure(figsize=(13, 5))
    plt.plot(fag1, Xag1_mag) # magnitude spectrum
    plt.xlim(xlima, xlimb)
    plt.xlabel('Frequency (Hz)')
    
def fourier_normalizado(senal, sr = 22050, xlima = 0, xlimb = 22050):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy  
    
    Xag1 = scipy.fft(senal)
    Xag1_mag = np.absolute(Xag1)
    fag1 = np.linspace(0, sr, len(Xag1_mag))
    
    plt.figure(figsize=(13, 5))
    plt.plot(fag1, Xag1_mag/Xag1_mag.max()) # magnitude spectrum
    plt.xlim(xlima, xlimb)
    plt.xlabel('Frequency (Hz)')
    
def fourier_abs(senal, sr=22050):
    
    import numpy as np
    import scipy  
    
    Xag1 = scipy.fft(senal)
    Xag1_mag = np.absolute(Xag1)
    fag1 = np.linspace(0, sr, len(Xag1_mag))
    
    return Xag1_mag, fag1


def marcas(rmse, flatness):
    import numpy as np
    
    rmse_lim = rmse/rmse.max()
    flatness_lim = flatness/flatness.max()
    limite_rmse = 0.02
    limite_flat = 0.01
    
    vector = np.array([])
    for i, (r, f) in enumerate(zip(rmse_lim, flatness_lim)):
        if r < limite_rmse:
            if f > limite_flat:
                vector = np.append(vector, 0)
            else:
                vector = np.append(vector, 1)
        else:
            vector = np.append(vector, 1)
    
    return(vector)
            