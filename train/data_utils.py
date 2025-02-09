import numpy as np

def making_glitch_datasets(glitch_set, noise_set, glitch_loc):
    if glitch_loc == 'H':
        return np.concatenate((glitch_set.reshape(-1,1,101), noise_set.reshape(-1,1,101)), axis = 1)
    elif glitch_loc == 'L':
        return np.concatenate((noise_set.reshape(-1,1,101), glitch_set.reshape(-1,1,101)), axis = 1)
    
    
def making_training_sets():
    # load from competition dataset
    dataDir = "../../../../Data_cached"
    list_dataset = ['glitch_L', 'glitch_H', 'noise_L', 'noise_H', 'BBH_L', 'BBH_H', 'SGLF_L', 'SGLF_H']
    dataset = {}
    dataset_fft = {}
    dataset_final = {}
    
    dataset['glitch_L'] = np.load(dataDir+"/real_glitches_snrlt5_60132_4000Hz_25ms.npz")["strain_time_data"][:10000];
    dataset['glitch_H'] = np.load(dataDir+"/real_glitches_H_snrlt5_59732_4000Hz_25ms.npz")["strain_time_data"][:10000];
    dataset['noise_L'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data'][:30000,1,:]
    dataset['noise_H'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data'][:30000,0,:]
    dataset['BBH_L'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/bbh_for_challenge.npy')[:20000,1,:]
    dataset['BBH_H'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/bbh_for_challenge.npy')[:20000,0,:]
    dataset['SGLF_L'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/sglf_for_challenge.npy')[:20000,1,:]
    dataset['SGLF_H'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/sglf_for_challenge.npy')[:20000,0,:]

    for ds in list_dataset:
        np.random.shuffle(dataset[ds])

    for ds in list_dataset:
        dataset[ds] /= np.linalg.norm([dataset[ds]], axis=2).T
        dataset_fft[ds] = abs(np.fft.rfft(dataset[ds]))
        dataset_fft[ds] /= np.linalg.norm([dataset_fft[ds]], axis=2).T
    
    dataset_final[0] = making_glitch_datasets(dataset_fft['glitch_H'], dataset_fft['noise_L'][:10000], 'L')
    dataset_final[1] = making_glitch_datasets(dataset_fft['glitch_L'], dataset_fft['noise_H'][:10000], 'H')
    dataset_final[2] = making_glitch_datasets(dataset_fft['noise_H'][10000:], dataset_fft['noise_L'][10000:], 'H')
    dataset_final[3] = making_glitch_datasets(dataset_fft['BBH_H'], dataset_fft['BBH_L'], 'H')
    dataset_final[4] = making_glitch_datasets(dataset_fft['SGLF_H'], dataset_fft['SGLF_L'], 'H')
    
    return dataset_final
    
    
def making_testing_sets():
    # load from competition dataset
    dataDir = "../../../../Data_cached"
    list_dataset = ['glitch_L', 'glitch_H', 'noise_L', 'noise_H', 'BBH_L', 'BBH_H', 'SGLF_L', 'SGLF_H']
    dataset = {}
    dataset_fft = {}
    dataset_final = {}
    
    dataset['glitch_L'] = np.load(dataDir+"/real_glitches_snrlt5_60132_4000Hz_25ms.npz")["strain_time_data"][-1250:];
    dataset['glitch_H'] = np.load(dataDir+"/real_glitches_H_snrlt5_59732_4000Hz_25ms.npz")["strain_time_data"][-1250:];
    dataset['noise_L'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data'][-33750:,1,:]
    dataset['noise_H'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/background.npz')['data'][-33750:,0,:]
    dataset['BBH_L'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/bbh_for_challenge.npy')[-5000:,1,:]
    dataset['BBH_H'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/bbh_for_challenge.npy')[-5000:,0,:]
    dataset['SGLF_L'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/sglf_for_challenge.npy')[-5000:,1,:]
    dataset['SGLF_H'] = np.load('/mnt/GWNMMAD_data/Tw_dataset/Datasets/sglf_for_challenge.npy')[-5000:,0,:]

    for ds in list_dataset:
        np.random.shuffle(dataset[ds])

    for ds in list_dataset:
        dataset[ds] /= np.linalg.norm([dataset[ds]], axis=2).T
        dataset_fft[ds] = abs(np.fft.rfft(dataset[ds]))
        dataset_fft[ds] /= np.linalg.norm([dataset_fft[ds]], axis=2).T
    
    dataset_final[0] = making_glitch_datasets(dataset_fft['glitch_H'], dataset_fft['noise_L'][:1250], 'L')
    dataset_final[1] = making_glitch_datasets(dataset_fft['glitch_L'], dataset_fft['noise_H'][:1250], 'H')
    dataset_final[2] = making_glitch_datasets(dataset_fft['noise_H'][1250:], dataset_fft['noise_L'][1250:], 'H')
    dataset_final[2] = making_glitch_datasets(dataset_fft['BBH_H'], dataset_fft['BBH_L'], 'H')
    dataset_final[3] = making_glitch_datasets(dataset_fft['SGLF_H'], dataset_fft['SGLF_L'], 'H')
    
    dataset_wsl_fft_all = np.empty((0,202))

    for key in dataset_final.keys():
        dataset_wsl_fft_all = np.append(dataset_wsl_fft_all, dataset_final[key], axis = 0)
    
    return dataset_final
    