'''
This file analyzes behavior clusters using tSNE across all learnlight trials, across all animals.
Used to generate manuscript figures
'''

# from Import_pose_module import * ## SOMETHING IN HERE IS REMOVING ALL THE TICKS
# from PyPDF2 import PdfFileMerger, PdfFileReader
import matplotlib.cm as cm
import math
from matplotlib import animation
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mtp
import os, sys, glob
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import FastICA
import pdb
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.cluster import KMeans
# flash_detection_folder = '/media/tzhang/Tony_WD_4TB/LearnLight/LearnLight_LEDandNosepokeFlashes_Extracted'
# pose_folder = '/media/tzhang/Tony_WD_4TB/LearnLight/output'
#
# flash_files = glob.glob(flash_detection_folder+'/*LED_and_nosepoke_flashes.npy')
# flash_files = np.sort(np.array([os.path.basename(file) for file in flash_files]))
# pose_files = glob.glob(pose_folder+'/*.mj2_pose.json')
# pose_files = np.sort(np.array([os.path.basename(file) for file in pose_files]))

def process_raw_flashes(nosepoke_flashes_raw, trialtype_raw):
    '''
    This function takes a list of frames that are above threshold and outputs the frame
    that's the very last time stamp at which the stimulus (nosepoke hole) is flashing
    :param nosepoke_flashes_raw: all the raw nosepoke flash frames
    :param trialtype_raw: corresponding information about Left or Right nosepoke port
    :return:
    processed_trial_types: 0 (L) and 1 (R)
    '''
    assert np.size(nosepoke_flashes_raw) == np.size(trialtype_raw), 'Mismatch in length'
    processed_stim_times = [] # after nosepoke_flashes processed
    processed_trial_types = []
    converted_trial_types = []
    for i in range(1,len(nosepoke_flashes_raw)):
        current_frame, prev_frame = nosepoke_flashes_raw[i], nosepoke_flashes_raw[i-1]
        if current_frame != prev_frame + 1:
            prev_trial = trialtype_raw[i-1]
            processed_stim_times.append(prev_frame)
            processed_trial_types.append(prev_trial)
        elif i == len(nosepoke_flashes_raw) - 1:
            # if reaching last item in list, append it.
            current_trial = trialtype_raw[i]
            processed_stim_times.append(current_frame)
            processed_trial_types.append(current_trial)
        # remap 'L' and 'R' into 0s and 1s
    for _, type in enumerate(processed_trial_types): # remap
        if type == 'L':
            converted_trial_types.append(0)
        else:
            converted_trial_types.append(1)
    return (np.array(processed_stim_times), np.array(converted_trial_types))

def process_data():
    data, embedding_each_sess, all_startframes, all_trialtypes = {}, {}, {}, {} # all the preprocessed data from all files before computing tSNE
    for i in range(np.size(flash_files)):
        Pose = Load_Pose(pose_folder + '/' + pose_files[i])
        smoothed = Smooth_Keypoints(Pose, window=10)
        smoothed_keypoints = smoothed.Weighted_Mov_Avg()
        Post_smooth_analysis = Analysis(smoothed_keypoints, smoothed, pose_files[i])
        centroids = Post_smooth_analysis.compute_centroid()
        # extract info from flash times
        Flash = np.load(flash_detection_folder+'/'+flash_files[i])[()]
        nosepoke_flashes_raw = Flash['nosepoke_flashes']
        trialtype_raw = Flash['trial_types']
        trialstart_frames, trial_TYPE = process_raw_flashes(nosepoke_flashes_raw, trialtype_raw)
        # plot and analyze local embeddings
        Post_smooth_analysis.tSNE_test_intertrial(100, trialstart_frames, trial_TYPE,
                                                  padding=True, label=True, save = True)
        data[pose_files[i]] = Post_smooth_analysis.tSNE_data
        embedding_each_sess[pose_files[i]] = Post_smooth_analysis.embedded
        all_startframes[pose_files[i]] = Post_smooth_analysis.trial_history
        all_trialtypes[pose_files[i]] = Post_smooth_analysis.trial_TYPE
        # update / save data and embedding files
        np.save('learnlight_tSNE_preprocessed_data.npy', data)
        np.save('learnlight_tSNE_individual_session_embeddings.npy', embedding_each_sess)
        np.save('learnlight_tSNE_all_startframes.npy', all_startframes)
        np.save('learnlight_tSNE_all_trialtypes.npy', all_trialtypes)

def process_reward_info(csv_dir, all_trialtypes):
    '''
    import mu's csv files
    all_trialtypes: trial information obtained directly from videos
    '''
    files = glob.glob(csv_dir+'/*.csv')
    files.sort() # sort by animal 1 - 4
    reward_info = {}

    for i in files: # choose one animal
        csvimport = np.loadtxt(i, delimiter=",", skiprows = 1)[:,:4]
        '''
        1st 4 columns in following order: Episode, TrialTypes, Choices, Rewards
        '''
        _, sessionidxs = np.unique(csvimport[:,0], return_index = True)
        sessionidxs = np.append(sessionidxs, csvimport.shape[0]) # add idx of last trial

        all_sessions = list(Data_for_tSNE.keys())
        all_sess_1animal = [sess for sess in all_sessions if
                            sess[6] == i[-20]]  # names of sessions for current animal number
        # remove sessions that didn't last 90 trials
        all_sess_1animal = [s for s in all_sess_1animal if len(all_trialtypes[s]) > 80]
        print(len(all_sess_1animal))

        assert len(all_sess_1animal) == len(sessionidxs)-1, 'Error: session length mismatch. File: ' + str(i)

        for j in range(len(sessionidxs)-1): # pick one session

            # IF RUNNING CODE ON SINGLE SESSION, START HERE
            j = 16
            ### select the correct file name
            sess_j_name = all_sess_1animal[j]

            j = 15# delete later

            startidx = sessionidxs[j]
            endidx = sessionidxs[j+1]
            trialtypes_j = csvimport[startidx:endidx,1]
            reward_j = csvimport[startidx:endidx,2:] # choice, reward each occupying one column



            trialtypes_j_video = all_trialtypes[sess_j_name]

            assert(len(trialtypes_j) >= len(trialtypes_j_video))

            ### sliding window match with truncated trial type info obtained from nosepoke flashes from video!
            matched = False
            for w in range(len(trialtypes_j) - len(trialtypes_j_video)+1):
                window = trialtypes_j[w:w+len(trialtypes_j_video)]
                if all(window == trialtypes_j_video):
                    print('Session: '+sess_j_name+'.. MATCHED')
                    ### save into new files: choice (1st column), reward vs no reward (2nd column)
                    reward_j_window = reward_j[w:w+len(trialtypes_j_video)]
                    reward_info[sess_j_name] = reward_j_window
                    matched = True
                    break
                # check for missing trials from video flash detection
            if matched == False:
                print('Session: '+sess_j_name+'.. MATCH FAILED')
                ## manual mode kick in:
                print('Video    Matfile')
                print(np.concatenate((np.indices((len(trialtypes_j_video),1))[0],
                                np.reshape(trialtypes_j_video, (len(trialtypes_j_video), 1)),
                                np.reshape(trialtypes_j[0:len(trialtypes_j_video)], (len(trialtypes_j_video), 1))), 1))
                skip_trial_nbs_idx0 = input('Automated matching failed. Input trials from matfile for skipping, separated by space: ')
                skip_list = [int(i) for i in skip_trial_nbs_idx0.split(' ') if i.isdigit()]
                print('Deleting trials (0 indexed): '+str(skip_list))
                #truncate rewards
                truncated_trialtypes = [x for z,x in enumerate(trialtypes_j) if z not in skip_list]
                reward_j_truncated = [x for z,x in enumerate(reward_j) if z not in skip_list]
                ########## redo matching
                matched = False
                for w in range(len(truncated_trialtypes) - len(trialtypes_j_video) + 1):
                    window = truncated_trialtypes[w:w + len(trialtypes_j_video)]
                    if all(window == trialtypes_j_video):
                        print('Session: ' + sess_j_name + '.. MANUAL MATCHING SUCCESSFUL')
                        ### save into new files: choice (1st column), reward vs no reward (2nd column)
                        reward_j_window = reward_j_truncated[w:w + len(trialtypes_j_video)]
                        reward_info[sess_j_name] = reward_j_window
                        matched = True
                        break
                if matched == False:
                    print('MANUAL MATCH FAILED.')
            np.save('learnlight_tSNE_all_rewardinfo.npy', reward_info)


## INITIALIZATION

## mac os

# reward_info = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/learnlight_tSNE_all_rewardinfo.npy')[()]
# #### Combine tSNE features together for each mouse ###
# Data_for_tSNE = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/learnlight_tSNE_preprocessed_data.npy')[()]
# all_startframes = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/learnlight_tSNE_all_startframes.npy')[()]
# all_trialtypes = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/learnlight_tSNE_all_trialtypes.npy')[()]
reward_info = np.load('learnlight_tSNE_all_rewardinfo.npy')[()]
#### Combine tSNE features together for each mouse ###
Data_for_tSNE = np.load('learnlight_tSNE_preprocessed_data.npy')[()]
all_startframes = np.load('learnlight_tSNE_all_startframes.npy')[()]
all_trialtypes = np.load('learnlight_tSNE_all_trialtypes.npy')[()]


# select out animal-specific data
all_sessions = list(reward_info.keys()) # use only trials in reward info (mu's csv)
all_sessions.sort()


def combine_tSNE_features(animal, all_sessions):
    all_sessions.sort()
    print('Preparing tSNE features for animal '+str(animal)+'..')
    combined_features = [] # features from each session / trial combined together
    combined_startframes = [] # startframes of each trial across all sessions picked
    combined_trialtypes = [] # left or right. 0 or 1.
    combined_rewardinfo = []
    corresponding_sess = [] # names of all the sessions corresponding to each

    for sess in all_sessions:
        '''
        format: splitting tSNE features into each animal.
        '''
        if int(sess[6]) == animal:
            # print('Session: '+sess)
            data_sess = Data_for_tSNE[sess]
            startframes_sess = all_startframes[sess]
            trialtypes_sess = all_trialtypes[sess]
            rewardinfo_sess = reward_info[sess]
            nb_trials = np.shape(data_sess)[0]

            corresponding_sess.extend(nb_trials * [sess])
            # print(np.shape(nb_trials * [sess]))
            combined_features.extend(data_sess)
            # print(np.shape(data_sess))
            combined_startframes.extend(startframes_sess)
            # print(np.shape(startframes_sess))
            combined_trialtypes.extend(trialtypes_sess)
            # print(np.shape(trialtypes_sess))
            combined_rewardinfo.extend(rewardinfo_sess)
    print()

    return combined_features, corresponding_sess, combined_startframes, combined_trialtypes, combined_rewardinfo

## compute tSNE

def compute_colormap(combined_trialtypes, combined_rewardinfo):
    assert np.shape(combined_trialtypes)[0] == np.shape(combined_rewardinfo)[0], 'length mismatch between combined reward and trialtype'
    colors = []
    markers = []
    subplotcolors = []
    # cmap = {(0,1): 'g', (0,-1): 'c', (1,1): 'r', (1,-1): 'm', (0,0): 'c', (1,0): 'm'}
    cmap = {(0,1): 'red', (0,-1): 'red', (1,1): 'blue', (1,-1): 'blue', (0,0): 'red', (1,0): 'blue'}
    mmap = {(0,1): 'o', (0,-1): 'x', (1,1): 'o', (1,-1): 'x', (0,0): 'x', (1,0): 'x'}
    ### MAP: REWARD = O, WRONG = X. LEFT STIM = RED, RIGHT STIM = BLUE.
    # for i in range(np.shape(combined_rewardinfo)[0]):
    #     trialtype_i = combined_trialtypes[i]
    #     reward_i = combined_rewardinfo[i][-1]
    #     colors.append(cmap[trialtype_i, reward_i])
    #     markers.append(mmap[trialtype_i, reward_i])
    # new colormap
    reds = cm.Reds(np.linspace(0,1,np.shape(combined_rewardinfo)[0]))
    blues = cm.Blues(np.linspace(0,1,np.shape(combined_rewardinfo)[0]))
    for i in range(np.shape(combined_rewardinfo)[0]):
        trialtype_i = combined_trialtypes[i]
        reward_i = combined_rewardinfo[i][-1]
        markers.append(mmap[trialtype_i, reward_i])
        color = cmap[trialtype_i, reward_i]
        subplotcolors.append(color)
        if color == 'blue':
            colors.append(blues[i])
        else:
            colors.append(reds[i])
    return (np.array(colors), np.array(markers), np.array(subplotcolors))

# combine behavior features for plotting against

def combine_features(animal, all_sessions):
    all_sessions.sort()
    print('Preparing 3D features for animal '+str(animal)+'..')
    combined_features = [] # features from each session / trial combined together
    combined_startframes = [] # startframes of each trial across all sessions picked
    combined_trialtypes = [] # left or right. 0 or 1.
    combined_rewardinfo = []
    corresponding_sess = [] # names of all the sessions corresponding to each

    for sess in all_sessions:
        '''
        format: splitting tSNE features into each animal.
        '''
        if int(sess[6]) == animal:
            # print('Session: '+sess)
            data_sess = Data_for_tSNE[sess]
            startframes_sess = all_startframes[sess]
            trialtypes_sess = all_trialtypes[sess]
            rewardinfo_sess = reward_info[sess]
            nb_trials = np.shape(data_sess)[0]

            corresponding_sess.append(nb_trials * [sess])
            # print(np.shape(nb_trials * [sess]))
            combined_features.append(data_sess)
            # print(np.shape(data_sess))
            combined_startframes.append(startframes_sess)
            # print(np.shape(startframes_sess))
            combined_trialtypes.append(trialtypes_sess)
            # print(np.shape(trialtypes_sess))
            combined_rewardinfo.append(rewardinfo_sess)
    print()
    return combined_features, corresponding_sess, combined_startframes, combined_trialtypes, combined_rewardinfo


# plot 3 behavior plots per trial

def compute_euc_dist(centroids, centerport = None, dimension = 'y', normalize = True):
    '''
    :param centroids: 100x2 centroid averages from a cluster
    :param centerport: 2x1 x and y of average starting position
    :return: distances between centroids and centerpoirt
    '''
    euc_dist = np.empty((np.shape(centroids)[0], np.shape(centroids)[1]))
    for trial_nb, trial in enumerate(centroids):
        if normalize:
            centerport = trial[0]
        for frame, pair in enumerate(trial):
            if dimension == 'y':
                dist = pair[1]- centerport[1]
                euc_dist[trial_nb,frame] = dist
            else:
                pair_dist = math.sqrt((centerport[0] - pair[0])**2 + (centerport[1] - pair[1])**2)
                sign = np.sign(pair[1] - centerport[1]) # defined by y point
                pair_dist = sign * pair_dist
                euc_dist[trial_nb,frame] = pair_dist
    return euc_dist

#### tSNE function ###

def compute_and_plot_tSNE(combined_features, combined_startframes, combined_trialtypes, combined_rewardinfo,
                          corresponding_sess, label = False, save = False, plot_per_session = True,
                          recompute_tSNE = False, plot_cluster_centroids = False, title = False):
    plt.ioff()
    ### create folder where all the output files will be saved
    folder_dir = 'tSNE_learnlight_animal'+str(animal)+'_window'+str(np.shape(combined_features)[1])+'/'
    embedding_filename = '/embedded_animal' + str(animal) + '.npy'
    if not os.path.exists(folder_dir) or recompute_tSNE:
        print('Computing tSNE...')
        tSNErun = TSNE(n_components=2, verbose=True, perplexity=50, n_iter=2000)
        truncated_features = np.array(combined_features)
        embedded = tSNErun.fit_transform(truncated_features)
        if not os.path.exists(folder_dir): os.mkdir(folder_dir)
        np.save(folder_dir + embedding_filename, embedded)
    else:
        print('Previous tSNE embedding file exists. Proceeding to plotting..')
        embedded = np.load(folder_dir+embedding_filename)
    print('Plotting all sessions..')
    # PLOT MAIN FIGURE
    plt.figure(figsize=(3.8, 3.8))
    if title:
        plt.title('LearnLight (Animal '+str(animal)+')')
    x_min, x_max, y_min, y_max = min(embedded[:, 0]) - 10, max(embedded[:, 0]) + 10, \
                                 min(embedded[:, 1]) - 10, max(embedded[:, 1]) + 10
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    # COMPUTE COLOR MAPS (4 COLORS. 0: LL, 1: LR, 2: RR, 3: RL)
    # colormap, markermap, subplotcolors = compute_colormap(combined_trialtypes, combined_rewardinfo)
    # plot main figure
    # WITH COLORS AND CROSSES
    # plt.scatter(embedded[markermap=='o', 0], embedded[markermap=='o', 1], cmap='bwr', marker = 'o',
    #             c=colormap[markermap=='o'], s=22, edgecolor='black', linewidths=0.4, label = 'Reward')
    # plt.scatter(embedded[markermap=='x', 0], embedded[markermap=='x', 1], cmap='bwr', marker = 'X',
    #             c=colormap[markermap=='x'], s=43, edgecolor='black', linewidths=0.4, label = 'Incorrect')
    # NEW COLOR SCHEME (JUST 4 COLORS, NO SHAPES)
    # reward: 1 = correct, -1 = incorrect, 0 = timeout
    combined_trialtypes = np.array(combined_trialtypes)
    combined_rewardinfo = np.array(combined_rewardinfo)
    ## LL
    idxs = np.intersect1d(np.where(combined_trialtypes == 0)[0],
                          np.where(combined_rewardinfo[:,-1] == 1)[0])
    plt.scatter(embedded[idxs,0], embedded[idxs,1], c = 'blue', s = 13,
                edgecolor = 'black', linewidths = 0.4, label = 'LL')
    ## RR
    idxs = np.intersect1d(np.where(combined_trialtypes == 1)[0],
                          np.where(combined_rewardinfo[:,-1] == 1)[0])
    plt.scatter(embedded[idxs,0], embedded[idxs,1], c = 'darkorange', s = 13,
                edgecolor = 'black', linewidths = 0.4, label = 'RR')
    ## LR
    idxs = np.intersect1d(np.where(combined_trialtypes == 0)[0],
                          np.union1d(np.where(combined_rewardinfo[:,-1] == -1)[0],
                                     np.where(combined_rewardinfo[:,-1] == 0)[0]))
    plt.scatter(embedded[idxs,0], embedded[idxs,1], c = 'darkgreen', s = 13,
                edgecolor = 'black', linewidths = 0.4, label = 'LR')
    ## RL
    idxs = np.intersect1d(np.where(combined_trialtypes == 1)[0],
                          np.union1d(np.where(combined_rewardinfo[:,-1] == -1)[0],
                                     np.where(combined_rewardinfo[:,-1] == 0)[0]))
    plt.scatter(embedded[idxs,0], embedded[idxs,1], c = 'red', s = 13,
                edgecolor = 'black', linewidths = 0.4, label = 'RL')
    # pdb.set_trace()
    # turn box into just axis
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # create legend
    # plt.legend(frameon = False)
    # red_patch = patches.Patch(color=cm.Reds(0.8), label='Left stimulus')
    # blue_patch = patches.Patch(color=cm.Blues(0.8), label='Right stimulus')
    # legend_markers = plt.legend(frameon= False, loc = 'lower right')
    # legend_colors = plt.legend(frameon=False, handles = [red_patch, blue_patch], loc = 'lower left')
    # plt.gca().add_artist(legend_markers)
    # plt.gca().add_artist(legend_colors)
    if label:
        for count, frame in enumerate(combined_startframes):
            plt.annotate(frame, (embedded[count, 0], embedded[count, 1]), fontsize=6)
            ''' note: this annotation is for the frames in the VIDEO, NOT indices of the centroids / orientations! 
            Remember to subtract by shift '''
    if save: # main figure
        plt.savefig(folder_dir+'learnlight_' + str(animal) + '.pdf',
                    bbox_inches='tight')
        plt.savefig(folder_dir+'learnlight_' + str(animal) + '.png', dpi = 500,
                    bbox_inches='tight')
        print('Main figure saved')
    if plot_per_session: # plotted per session on separate plots
        plt.ioff()
        '''
        plot the embeddings one session at a time
        '''
        print('Saving session figures..')
        unique_sessions, idxs = np.unique(corresponding_sess, return_index=True)
        idxs = np.append(idxs, len(corresponding_sess))
        # set x and y axis limits as to maintain proportion
        merge_list = [] # keep track of list of files for merging in pdf
        for i in range(len(unique_sessions)):
            plt.figure(figsize=(8, 7))
            plt.title(unique_sessions[i])
            plt.xlim(xmin=x_min, xmax=x_max)
            plt.ylim(ymin=y_min, ymax=y_max)
            start_idx = idxs[i]
            end_idx = idxs[i+1]
            x, y = embedded[start_idx:end_idx, 0], embedded[start_idx:end_idx, 1]
            colors = subplotcolors[start_idx:end_idx]
            markers = markermap[start_idx:end_idx]
            # plt.scatter(x, y, cmap='bwr', c=combined_trialtypes[start_idx:idxs[i+1]], s=25, edgecolor='black', linewidths=0.9)
            if label:
                for count, frame in enumerate(combined_startframes[start_idx:end_idx]):
                    plt.annotate(frame, (x[count], y[count]), fontsize=6)
                # old plotting function
                # plt.scatter(x, y, cmap='bwr', c=combined_trialtypes[start_idx:-1], s=25, edgecolor='black', linewidths=0.9)
            # PLOT HERE!
            plt.scatter(x[markers == 'o'], y[markers == 'o'], cmap='bwr', marker='o',
                        c=colors[markers == 'o'], s=25, edgecolor='black', linewidths=0.9)
            plt.scatter(x[markers == 'x'], y[markers == 'x'], cmap='bwr', marker='X',
                        c=colors[markers == 'x'], s=45, edgecolor='black', linewidths=0.9)

            if save:
                fig_name = 'tSNE_plots_' + unique_sessions[i][:-14] + '.pdf'
                merge_list.append(fig_name)
                plt.savefig(folder_dir+fig_name)
                print('Session figure saved: ' + unique_sessions[i][:-14])
        # merge pdfs
        merger = PdfFileMerger()
        for filename in merge_list:
            file = open(folder_dir+filename, 'rb')
            merger.append(PdfFileReader(file))
            file.close()
        merger.write(folder_dir + 'learnlight_merged_sessionplots_animal' + str(animal)+'.pdf')
    if plot_cluster_centroids:
        # plt.ioff()
        '''
        plot centroids of each type (LL, LR, RR, RL) onto one plot showing drift across sessions
        '''
        print('Computing and generating centroid figures..')
        unique_sessions, idxs = np.unique(corresponding_sess, return_index=True)
        idxs = np.append(idxs, len(corresponding_sess)-1)
        # set x and y axis limits as to maintain proportion
        plt.figure(figsize=(8, 7))
        plt.xlim(xmin=x_min, xmax=x_max)
        plt.ylim(ymin=y_min, ymax=y_max)
        fig_name = 'tSNE_centroids_drift_animal' + str(animal)
        plt.title(fig_name)
        centroids = np.zeros((len(unique_sessions), 2, 2, 4))  # (trial, reward, x/y)

        combined_rewardinfo = np.array(combined_rewardinfo)
        combined_trialtypes = np.array(combined_trialtypes)

        for i in range(len(unique_sessions)):
            start_idx = idxs[i]
            end_idx = idxs[i+1]
            x_sess, y_sess = embedded[start_idx:end_idx, 0], embedded[start_idx:end_idx, 1]
            trialtype_sess = combined_trialtypes[start_idx:end_idx]
            reward_sess = combined_rewardinfo[start_idx:end_idx,-1]
            # Compute centroids
            for trial in [0,1]: # 0 = left, 1 = right
                for r, reward in enumerate([-1,1]): # 0 = no reward, 1 = reward
                    sub_idxs = np.logical_and(trialtype_sess == trial, reward_sess == reward)
                    # pdb.set_trace()
                    x, y = x_sess[sub_idxs], y_sess[sub_idxs]
                    # x and y coordinates
                    centroids[i, trial, r, 0] = np.average(x) # centroid-x
                    centroids[i, trial, r, 1] = np.average(y) # centroid-y
                    # SD along x and y axis for plotting error bars
                    centroids[i, trial, r, 2] = np.std(x) # x-directional standard deviation
                    centroids[i, trial, r, 3] = np.std(y) # y-directinoal standard deviation
                    '''
                    note: if there isn't a single trial matching (trial, reward), each field would save a 'NaN'
                    '''
        # PLOT HERE!
        for trial in [0,1]: # left and right
            for reward in [0,1]: # reward and no reward
                x = centroids[:,trial,reward,0]
                y = centroids[:,trial,reward,1]
                x_sd = centroids[:,trial,reward,2]
                y_sd = centroids[:,trial,reward,3]
                # Filter out NaN values
                NaN_idxs = np.logical_or.reduce((np.isnan(x), np.isnan(y), np.isnan(x_sd), np.isnan(y_sd)))
                x, y, x_sd, y_sd = x[~NaN_idxs], y[~NaN_idxs], x_sd[~NaN_idxs], y_sd[~NaN_idxs]
                avg_sd = np.average([x_sd, y_sd], axis = 0)
                if reward:
                    marker = 'o'
                    # size = 30
                    size = avg_sd * 5 + 2
                else:
                    marker = 'X'
                    # size = 50
                    size = avg_sd * 5 + 17
                if trial == 0:
                    cmap = 'Reds'
                    linecolor = 'red'
                    gradient = cm.Reds(np.linspace(0,1,np.shape(x)[0]))
                else:
                    cmap = 'Blues'
                    linecolor = 'blue'
                    gradient = cm.Blues(np.linspace(0,1,np.shape(x)[0]))
                # PLOT

                plt.plot(x,y,color = linecolor, zorder=1, lw=1)
                plt.scatter(x, y, cmap=cmap, marker=marker,
                            c=gradient, s=size, edgecolor='black', linewidths=1, zorder=2, alpha = 0.8)
                # plt.savefig(folder_dir + fig_name + '.pdf')
                # pdb.set_trace()
        if save:
            plt.savefig(folder_dir+fig_name+'.pdf')
            print('Centroid figure saved: ' + fig_name)

#### PCA function ###

def compute_and_plot_PCA(combined_features, combined_startframes, combined_trialtypes, combined_rewardinfo,
                          corresponding_sess, label = False, save = False, plot_per_session = True,
                          recompute_PCA = False, plot_cluster_centroids = False, title = False):
    plt.ioff()
    ### create folder where all the output files will be saved
    folder_dir = 'PCA_learnlight_animal'+str(animal)+'_window'+str(np.shape(combined_features)[1]//3)+'/'
    embedding_filename = '/embedded_PCA_animal' + str(animal) + '.npy'
    if not os.path.exists(folder_dir) or recompute_PCA:
        print('Computing PCA...')
        PCArun = PCA(n_components=2, whiten = False)
        truncated_features = np.array(combined_features)
        embedded = PCArun.fit_transform(truncated_features)
        if not os.path.exists(folder_dir): os.mkdir(folder_dir)
        np.save(folder_dir + embedding_filename, embedded)
    else:
        print('Previous PCA embedding file exists. Proceeding to plotting..')
        embedded = np.load(folder_dir+embedding_filename)
    print('Plotting all sessions..')
    # PLOT MAIN FIGURE
    plt.figure(figsize=(5, 4.5))
    if title:
        plt.title('LearnLight (Animal '+str(animal)+')')
    x_min, x_max, y_min, y_max = min(embedded[:, 0]) - 100, max(embedded[:, 0]) + 100, \
                                 min(embedded[:, 1]) - 100, max(embedded[:, 1]) + 200
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    # COMPUTE COLOR MAPS (4 COLORS. 0: LL, 1: LR, 2: RR, 3: RL)
    colormap, markermap, subplotcolors = compute_colormap(combined_trialtypes, combined_rewardinfo)
    # plot main figure
    plt.scatter(embedded[markermap=='o', 0], embedded[markermap=='o', 1], cmap='bwr', marker = 'o',
                c=colormap[markermap=='o'], s=19, edgecolor='black', linewidths=0.5, label = 'Reward')
    plt.scatter(embedded[markermap=='x', 0], embedded[markermap=='x', 1], cmap='bwr', marker = 'X',
                c=colormap[markermap=='x'], s=37, edgecolor='black', linewidths=0.5, label = 'Incorrect')
    # create legend
    red_patch = patches.Patch(color=cm.Reds(0.8), label='Left stimulus')
    blue_patch = patches.Patch(color=cm.Blues(0.8), label='Right stimulus')
    legend_markers = plt.legend(frameon= False, loc = 'upper right') # upper vs lower
    legend_colors = plt.legend(frameon=False, handles = [red_patch, blue_patch], loc = 'upper left')
    plt.gca().add_artist(legend_markers)
    plt.gca().add_artist(legend_colors)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    if label:
        for count, frame in enumerate(combined_startframes):
            plt.annotate(frame, (embedded[count, 0], embedded[count, 1]), fontsize=6)
            ''' note: this annotation is for the frames in the VIDEO, NOT indices of the centroids / orientations! 
            Remember to subtract by shift '''
    if save: # main figure
        plt.savefig(folder_dir+'PCA_learnlight_' + str(animal) + '.pdf',
                    bbox_inches = 'tight')
        plt.savefig(folder_dir+'PCA_learnlight_' + str(animal) + '.png',
                    dpi = 500, bbox_inches = 'tight')
        print('Main figure saved')
    if plot_per_session: # plotted per session on separate plots
        plt.ioff()
        '''
        plot the embeddings one session at a time
        '''
        print('Saving session figures..')
        unique_sessions, idxs = np.unique(corresponding_sess, return_index=True)
        idxs = np.append(idxs, len(corresponding_sess))
        # set x and y axis limits as to maintain proportion
        merge_list = [] # keep track of list of files for merging in pdf
        for i in range(len(unique_sessions)):
            plt.figure(figsize=(8, 7))
            plt.title(unique_sessions[i])
            plt.xlim(xmin=x_min, xmax=x_max)
            plt.ylim(ymin=y_min, ymax=y_max)
            start_idx = idxs[i]
            end_idx = idxs[i+1]
            x, y = embedded[start_idx:end_idx, 0], embedded[start_idx:end_idx, 1]
            colors = subplotcolors[start_idx:end_idx]
            markers = markermap[start_idx:end_idx]
            # plt.scatter(x, y, cmap='bwr', c=combined_trialtypes[start_idx:idxs[i+1]], s=25, edgecolor='black', linewidths=0.9)
            if label:
                for count, frame in enumerate(combined_startframes[start_idx:end_idx]):
                    plt.annotate(frame, (x[count], y[count]), fontsize=6)
                # old plotting function
                # plt.scatter(x, y, cmap='bwr', c=combined_trialtypes[start_idx:-1], s=25, edgecolor='black', linewidths=0.9)
            # PLOT HERE!
            plt.scatter(x[markers == 'o'], y[markers == 'o'], cmap='bwr', marker='o',
                        c=colors[markers == 'o'], s=25, edgecolor='black', linewidths=0.9)
            plt.scatter(x[markers == 'x'], y[markers == 'x'], cmap='bwr', marker='X',
                        c=colors[markers == 'x'], s=45, edgecolor='black', linewidths=0.9)

            if save:
                fig_name = 'PCA_plots_' + unique_sessions[i][:-14] + '.pdf'
                merge_list.append(fig_name)
                plt.savefig(folder_dir+fig_name)
                print('Session figure saved: ' + unique_sessions[i][:-14])
        # merge pdfs
        merger = PdfFileMerger()
        for filename in merge_list:
            file = open(folder_dir+filename, 'rb')
            merger.append(PdfFileReader(file))
            file.close()
        merger.write(folder_dir + 'PCA_learnlight_merged_sessionplots_animal' + str(animal)+'.pdf')
    if plot_cluster_centroids:
        # plt.ioff()
        '''
        plot centroids of each type (LL, LR, RR, RL) onto one plot showing drift across sessions
        '''
        print('Computing and generating centroid figures..')
        unique_sessions, idxs = np.unique(corresponding_sess, return_index=True)
        idxs = np.append(idxs, len(corresponding_sess)-1)
        # set x and y axis limits as to maintain proportion
        plt.figure(figsize=(8, 7))
        plt.xlim(xmin=x_min, xmax=x_max)
        plt.ylim(ymin=y_min, ymax=y_max)
        fig_name = 'PCA_centroids_drift_animal' + str(animal)
        plt.title(fig_name)
        centroids = np.zeros((len(unique_sessions), 2, 2, 4))  # (trial, reward, x/y)

        combined_rewardinfo = np.array(combined_rewardinfo)
        combined_trialtypes = np.array(combined_trialtypes)

        for i in range(len(unique_sessions)):
            start_idx = idxs[i]
            end_idx = idxs[i+1]
            x_sess, y_sess = embedded[start_idx:end_idx, 0], embedded[start_idx:end_idx, 1]
            trialtype_sess = combined_trialtypes[start_idx:end_idx]
            reward_sess = combined_rewardinfo[start_idx:end_idx,-1]
            # Compute centroids
            for trial in [0,1]: # 0 = left, 1 = right
                for r, reward in enumerate([-1,1]): # 0 = no reward, 1 = reward
                    sub_idxs = np.logical_and(trialtype_sess == trial, reward_sess == reward)
                    # pdb.set_trace()
                    x, y = x_sess[sub_idxs], y_sess[sub_idxs]
                    # x and y coordinates
                    centroids[i, trial, r, 0] = np.average(x) # centroid-x
                    centroids[i, trial, r, 1] = np.average(y) # centroid-y
                    # SD along x and y axis for plotting error bars
                    centroids[i, trial, r, 2] = np.std(x) # x-directional standard deviation
                    centroids[i, trial, r, 3] = np.std(y) # y-directinoal standard deviation
                    '''
                    note: if there isn't a single trial matching (trial, reward), each field would save a 'NaN'
                    '''
        # PLOT HERE!
        for trial in [0,1]: # left and right
            for reward in [0,1]: # reward and no reward
                x = centroids[:,trial,reward,0]
                y = centroids[:,trial,reward,1]
                x_sd = centroids[:,trial,reward,2]
                y_sd = centroids[:,trial,reward,3]
                # Filter out NaN values
                NaN_idxs = np.logical_or.reduce((np.isnan(x), np.isnan(y), np.isnan(x_sd), np.isnan(y_sd)))
                x, y, x_sd, y_sd = x[~NaN_idxs], y[~NaN_idxs], x_sd[~NaN_idxs], y_sd[~NaN_idxs]
                avg_sd = np.average([x_sd, y_sd], axis = 0)
                if reward:
                    marker = 'o'
                    # size = 30
                    size = avg_sd * 5 + 2
                else:
                    marker = 'X'
                    # size = 50
                    size = avg_sd * 5 + 17
                if trial == 0:
                    cmap = 'Reds'
                    linecolor = 'red'
                    gradient = cm.Reds(np.linspace(0,1,np.shape(x)[0]))
                else:
                    cmap = 'Blues'
                    linecolor = 'blue'
                    gradient = cm.Blues(np.linspace(0,1,np.shape(x)[0]))
                # PLOT

                plt.plot(x,y,color = linecolor, zorder=1, lw=1)
                plt.scatter(x, y, cmap=cmap, marker=marker,
                            c=gradient, s=size, edgecolor='black', linewidths=1, zorder=2, alpha = 0.8)
                # plt.savefig(folder_dir + fig_name + '.pdf')
                # pdb.set_trace()
        if save:
            plt.savefig(folder_dir+fig_name+'.pdf')
            print('Centroid figure saved: ' + fig_name)


def Hierarchical_Clustering(combined_features, combined_startframes, combined_trialtypes, combined_rewardinfo,
                          corresponding_sess, label = False, save = False, plot_per_session = True,
                          recompute_PCA = False, plot_cluster_centroids = False, title = False):
    plt.ioff()
    for animal in [1,2,3,4]:
        for nb_clusters in [2,3,4,5,6]:
            combined_features, corresponding_sess, \
            combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.cluster.hierarchy import fcluster
            nb_frames = 45
            centroids = np.array(combined_features)[:, 0:2 * nb_frames]
            orientations = np.array(combined_features)[:, 200:200 + nb_frames]
            truncated_combined_features = np.concatenate((centroids, orientations), axis=1)
            Z = linkage(truncated_combined_features, 'ward')
            # nb_clusters = 2
            clusters = fcluster(Z, nb_clusters, criterion='maxclust')
            cluster_idxs = {} # dictionary object
            for cluster in np.unique(clusters):
                cluster_label = str(cluster)
                cluster_idxs[cluster_label] = np.where(clusters == cluster)[0]
            # plot hierarchical labels imposed onto tSNE plots
            embedding = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                                'tSNE_learnlight_animal'+str(animal)+
                                '_window135/embedded_animal'+str(animal)+'.npy')
            fig, ax = plt.subplots(figsize = (5,5))
            for cluster in list(cluster_idxs.keys()):
                idxs = cluster_idxs[cluster]
                ax.scatter(embedding[idxs,0],embedding[idxs,1],
                           s = 8, edgecolor = 'black', linewidths = 0.3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.savefig('tSNE_hierarchical_anim'+str(animal)+'nbclust'
                        + str(nb_clusters)+'.png', dpi = 300)
        plt.close('all')


    # plot distance plots
    plot_dist_to_nosepoke(animal, cluster_idxs=cluster_idxs, mode='position',
                          feature='centroids',
                          error='SD', separate_traces=False)


############### DISTANCE plots ##########################3

def compute_euc_dist(centroids, centerport, dimension = 'y', normalize = True):
    '''
    :param centroids: 100x2 centroid averages from a cluster
    :param centerport: 2x1 x and y of average starting position
    :return: distances between centroids and centerpoirt
    '''
    euc_dist = np.empty((np.shape(centroids)[0], np.shape(centroids)[1]))
    for trial_nb, trial in enumerate(centroids):
        if normalize:
            centerport = trial[0]
        for frame, pair in enumerate(trial):
            if dimension == 'y':
                dist = pair[1]- centerport[1]
                euc_dist[trial_nb,frame] = dist
            else:
                pair_dist = math.sqrt((centerport[0] - pair[0])**2 + (centerport[1] - pair[1])**2)
                sign = np.sign(pair[1] - centerport[1]) # defined by y point
                pair_dist = sign * pair_dist
                euc_dist[trial_nb,frame] = pair_dist
    return euc_dist

def compute_speed(centroids, dimension = 'y'):
    '''
    :param centroids: n*100x2 centroids from a cluster, where n is the number of trajectories
    :return: speeds in numpy form, in the dimension n x 99 since we're computing speeds between positions
    '''
    all_speeds = np.empty((np.shape(centroids)[0], np.shape(centroids)[1]-1))
    for n in range(np.shape(centroids)[0]):
        for i in range(np.shape(centroids)[1]-1):
            centroid_t1 = centroids[n, i]
            centroid_t2 = centroids[n, i+1]
            # speed is just distance per timestep
            if dimension == 'y':
                speed = centroid_t2[1] - centroid_t1[1]
            else:
                speed = math.sqrt((centroid_t1[0] - centroid_t2[0])**2 + (centroid_t1[1] - centroid_t2[1])**2)
                sign = np.sign(centroid_t2[1] - centroid_t1[1]) # defined by y point movement direction
                speed = sign * speed
            all_speeds[n,i] = speed
    return all_speeds

def compute_cumulative_dist(centroids, centerport):
    '''
    Compute distance beteen all points.
    :param centroids:
    :param centerport:
    :return:
    '''
    # for i in range(np.shape(centroids)[0])

def compute_ang_vel(orientations):
    '''
    :param orientation: n*100 angular information from a cluster, where n is the number of trajectories
    :return: speeds in numpy form, in the dimension n x 99 since we're computing speeds between positions
    '''
    all_angvel = np.empty((np.shape(orientations)[0], np.shape(orientations)[1]-1))
    for n in range(np.shape(orientations)[0]):
        for i in range(np.shape(orientations)[1]-1):
            orientation_t1 = orientations[n, i]
            orientation_t2 = orientations[n, i+1]
            # angular velocity is just distance per timestep
            angvel = orientation_t2 - orientation_t1
            all_angvel[n,i] = angvel
    return all_angvel


def plot_dist_to_nosepoke(animal, clusters_coord = None, embedding = None, cluster_idxs = None, mode = 'position',
                          feature = 'centroids', split = False, figure_title = False, error = 'SD',
                          separate_traces = False, use_cluster_coord = False, create_Legend = True,
                          custom_filename = None):
    '''
    :param animal: 1,2,3, or 4
    :param clusters_coord: the coordinates for the cluster
    :param embedding: embedding
    :param mode: velocity or position
    :param feature: centroids or orientation
    :param split: True = split into early, mid, and late stage training
    :param error: 'SE' or 'SD'
    use_cluster_coord: if False, use trial and reward information for clustering
    instead of clusters from tSNE / PCA
    '''
    flipaxis = True # video processing is flipped. Flip y-axis by * -1
    combined_features, corresponding_sess, combined_startframes, \
    combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    # NOTE: DECIDE IF SHOULD SPLIT BY TRIAL, OR BY SESSION??
    # if split: # SPLIT BY TRIAL
    #     # split into early, mid, and late training plots (3)
    #     group_size = round(np.size(corresponding_sess) / 3)
    #     split_3groups_idxs = np.arange()
    if clusters_coord is not None:
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]
    movement_metric = {}
    error_bars = {}
    combined_features = np.array(combined_features)
    avg_start_position = np.average(combined_features[:,:2], axis = 0)
    center_port = avg_start_position
    # determine groups for each cluster
    if use_cluster_coord:
        cluster_types = list(clusters_coord.keys())
    elif cluster_idxs is not None:
        cluster_types = list(cluster_idxs.keys())
    else:
        cluster_types = ['LL','LR','RR','RL']

    for cluster_type in cluster_types:
        if use_cluster_coord:
            cluster = clusters_coord[cluster_type]
            idxs = np.array([], dtype = 'int')
            for block in cluster:
                x1, x2, y1, y2 = block[0], block[1], block[2], block[3]
                ## tally up all points
                idxs = np.append(idxs, np.where((embedding_x >= x1) & (embedding_x <= x2) &
                                (embedding_y >= y1) & (embedding_y <= y2))[0])
        elif cluster_idxs is not None:
            idxs = cluster_idxs[cluster_type]
        else: # rely on trial and reward info for groups
            if cluster_type == 'LL':
                idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 0),
                                      np.where(np.array(combined_rewardinfo)[:,1] == 1))
            elif cluster_type == 'LR':
                idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 0),
                                      np.where(np.array(combined_rewardinfo)[:, 1] == -1))
            elif cluster_type == 'RR':
                idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 1),
                                      np.where(np.array(combined_rewardinfo)[:, 1] == 1))
            elif cluster_type == 'RL':
                idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 1),
                                      np.where(np.array(combined_rewardinfo)[:, 1] == -1))
        # pdb.set_trace()
        cluster_features = np.array(combined_features)[idxs]
        if split:
            # split cluster_features into 3 groups
            nb_features = np.shape(cluster_features)[0]
            features_per_group = round(nb_features/3)
            split_group_idxs = np.arange(0,nb_features,features_per_group)[:3]
            split_group_idxs = np.append(split_group_idxs, nb_features) # add length to indicate end of last idxs
            for group_nb in range(3):
                start_idx = split_group_idxs[group_nb]
                end_idx = split_group_idxs[group_nb+1]
                cluster_feature_split = cluster_features[start_idx:end_idx]
                ## old code
                features_avg = np.average(cluster_feature_split, axis=0)
                if feature == 'centroids':
                    features_avg = np.reshape(features_avg[:200], (1, 100, 2))
                    reshaped_features = np.reshape(cluster_feature_split[:, :200], (np.shape(cluster_feature_split)[0], 100, 2))
                    if mode == 'position':
                        movement_metric[cluster_type+str(group_nb)] = compute_euc_dist(features_avg, center_port)
                        cluster_feature_distance = compute_euc_dist(reshaped_features)
                        error_bars[cluster_type+str(group_nb)] = np.std(cluster_feature_distance)
                    if mode == 'velocity':
                        movement_metric[cluster_type+str(group_nb)] = compute_speed(features_avg)[0]
                        cluster_feature_speeds = compute_speed(reshaped_features)
                        error_bars[cluster_type+str(group_nb)] = np.std(cluster_feature_speeds, axis=0)
                if feature == 'orientation':
                    features_avg = np.reshape(features_avg[200:], (1, 100))
                    reshaped_features = np.reshape(cluster_feature_split[:, 200:], (np.shape(cluster_feature_split)[0], 100))
                    if mode == 'position':
                        movement_metric[cluster_type+str(group_nb)] = features_avg[0]
                        error_bars[cluster_type+str(group_nb)] = np.std(reshaped_features, axis=0)
                    if mode == 'velocity':
                        movement_metric[cluster_type+str(group_nb)] = compute_ang_vel(features_avg)[0]
                        cluster_feature_ang_vel = compute_ang_vel(reshaped_features)
                        error_bars[cluster_type+str(group_nb)] = np.std(cluster_feature_ang_vel, axis=0)
        else: # NO SPLITTING
            features_avg = np.average(cluster_features, axis=0)
            if feature == 'centroids':
                features_avg = np.reshape(features_avg[:200], (1,100,2))
                reshaped_features = np.reshape(cluster_features[:,:200],(np.shape(cluster_features)[0], 100, 2))
                if mode == 'position':
                    if separate_traces:
                        movement_metric[cluster_type] = compute_euc_dist(reshaped_features, center_port)
                    else:
                        movement_metric[cluster_type] = compute_euc_dist(features_avg, center_port)
                        cluster_feature_distance = compute_euc_dist(reshaped_features, center_port)
                        if error == 'SD':
                            error_bars[cluster_type] = np.std(cluster_feature_distance, axis=0)
                        elif error == 'SE':
                            error_bars[cluster_type] = stats.sem(cluster_feature_distance, axis=0)
                if mode == 'velocity':
                    if separate_traces:
                        movement_metric[cluster_type] = compute_speed(reshaped_features)
                    else:
                        movement_metric[cluster_type] = compute_speed(features_avg)[0]
                        cluster_feature_speeds = compute_speed(reshaped_features)
                        if error == 'SD':
                            error_bars[cluster_type] = np.std(cluster_feature_speeds, axis=0)
                        elif error == 'SE':
                            error_bars[cluster_type] = stats.sem(cluster_feature_speeds, axis=0)
            if feature == 'orientation':
                features_avg = np.reshape(features_avg[200:], (1,100))
                reshaped_features = np.reshape(cluster_features[:,200:],(np.shape(cluster_features)[0], 100))
                if mode == 'position':
                    if separate_traces:
                        movement_metric[cluster_type] = reshaped_features
                    else:
                        movement_metric[cluster_type] = features_avg[0]
                        if error == 'SD':
                            error_bars[cluster_type] = np.std(reshaped_features, axis=0)
                        elif error == 'SE':
                            error_bars[cluster_type] = stats.sem(reshaped_features, axis=0)
                if mode == 'velocity':
                    if separate_traces:
                        movement_metric[cluster_type] = compute_ang_vel(reshaped_features)
                    else:
                        movement_metric[cluster_type] = compute_ang_vel(features_avg)[0]
                        cluster_feature_ang_vel = compute_ang_vel(reshaped_features)
                        if error == 'SD':
                            error_bars[cluster_type] = np.std(cluster_feature_ang_vel, axis=0)
                        elif error == 'SE':
                            error_bars[cluster_type] = stats.sem(cluster_feature_ang_vel, axis=0)
    # PLOTTING
    if split:
        for group_nb in range(3):
            plt.figure(figsize=(3.7, 3.3))
            # set custom colors here
            col = {'1': 'blue',
                   '2': 'darkorange',
                   '3': 'darkgreen',
                   '4': 'red',
                   '5': 'cyan'}
            for cluster in list(clusters_coord.keys()):
                error_bars_top = movement_metric[cluster+str(group_nb)] + error_bars[cluster+str(group_nb)]
                error_bars_bottom = movement_metric[cluster+str(group_nb)] - error_bars[cluster+str(group_nb)]
                plt.fill_between(np.arange(45), -error_bars_bottom[:45], -error_bars_top[:45],
                                 interpolate=True, alpha=0.2,
                                 color = col[cluster])
                plt.plot(-1 * movement_metric[cluster+str(group_nb)][:45], label=cluster,
                         color = col[cluster])
                if feature == 'centroids':
                    plt.ylim(-14, 14)
                if feature == 'orientation':
                    if mode == 'position':
                        plt.ylim(-70, 70)
                    if mode == 'velocity':
                        plt.ylim(-6, 6)
                plt.xlim(0, 44)
                # plt.tight_layout()
            plt.legend()
            # SAVE PLOT
            if group_nb == 0: stage = 'EARLY'
            if group_nb == 1: stage = 'MID'
            if group_nb == 2: stage = 'LATE'
            if feature == 'centroids':
                if mode == 'velocity':
                    if figure_title:
                        plt.title(stage+' LearnLight Velocity Plot (Clusters) Animal ' + str(animal))
                    plt.savefig(stage+'_LearnLight_Velocity_Animal' + str(animal) + '.pdf')
                if mode == 'position':
                    if figure_title:
                        plt.title(stage+' LearnLight Distance Plot (Clusters) Animal ' + str(animal))
                    plt.savefig(stage+'_LearnLight_Distance_Animal' + str(animal) + '.pdf')
            if feature == 'orientation':
                if mode == 'position':
                    if figure_title:
                        plt.title(stage+' LearnLight Orientation Plot (Clusters) Animal ' + str(animal))
                    plt.savefig('LearnLight_Orientation_Animal' + str(animal) + '.pdf')
                if mode == 'velocity':
                    if figure_title:
                        plt.title(stage+' LearnLight Angular Velocity Plot (Clusters) Animal ' + str(animal))
                    plt.savefig(stage+'_LearnLight_AngularVelocity_Animal' + str(animal) + '.pdf')

    else: # NO SPLITTING
        if cluster_idxs is not None:
            # colormap = {'1': 'red', '2': 'blue', '3': 'orange', '4': 'green', '5': 'black'}
            # colormap = {'1': 'blue', '2': 'darkorange', '3': 'darkgreen', '4': 'red', '5': 'cyan'}
            colormap = {'1': 'C0', '2': 'C1', '3': 'C2', '4': 'C3', '5': 'cyan'}
            if animal == 2:
                colormap = {'1': 'C0', '2': 'C1', '3': 'C2', '4': 'C3', '5': 'C4', '6': 'cyan'}
            # colormap = {'LL': 'blue', 'RR': 'darkorange', 'LR': 'darkgreen', 'RL': 'red',
            #                 'Mixed (5th cluster)': 'cyan', 'Mixed (middle cluster)': 'yellow'}
            # # changed here!
        else:
            colormap = {'LL': 'blue', 'RR': 'darkorange', 'LR': 'darkgreen', 'RL': 'red',
                            'Mixed (5th cluster)': 'cyan', 'Mixed (middle cluster)': 'yellow'}
        plt.figure(figsize=(4.5,4))
        ax = plt.axes()
        for cluster in list(movement_metric.keys()):
            if separate_traces:
                print(np.shape(movement_metric[cluster][:,:45]))
                x = np.arange(0,45) # convert to seconds
                y = np.transpose(movement_metric[cluster][:,:45])
                color = colormap[cluster]
                if feature == 'centroids':
                    if mode == 'position':
                        plt.plot(x, -y, color = color, alpha = 0.1, linewidth = 0.3)
                    if mode == 'velocity':
                        plt.plot(x, -y, color=color, alpha=0.1, linewidth = 0.2)
                else:
                    if mode == 'position':
                        plt.plot(x, -y, color = color, alpha = 0.1, linewidth = 0.3)
                    if mode == 'velocity':
                        plt.plot(x, -y, color=color, alpha=0.1, linewidth = 0.2)
            else:
                # pdb.set_trace()
                color = colormap[cluster]
                if feature == 'orientation' or mode == 'velocity':
                    error_bars_top = movement_metric[cluster] + error_bars[cluster]
                    error_bars_bottom = movement_metric[cluster] - error_bars[cluster]
                    plt.fill_between(np.arange(45), -error_bars_bottom[:45], -error_bars_top[:45],
                                     interpolate=True, alpha=0.2, facecolor=color)
                    plt.plot(-movement_metric[cluster][0:45], label=cluster, color=color)
                else:
                    error_bars_top = movement_metric[cluster][0] + error_bars[cluster]
                    error_bars_bottom = movement_metric[cluster][0] - error_bars[cluster]
                    if color == 'cyan':
                        plt.fill_between(np.arange(45), -error_bars_bottom[:45], -error_bars_top[:45],
                                         interpolate=True, alpha=0.5, facecolor=color)
                        plt.plot(-movement_metric[cluster][0,0:45], label = cluster, color = 'teal',
                                 )
                    else:
                        plt.fill_between(np.arange(45), -error_bars_bottom[:45], -error_bars_top[:45],
                                         interpolate=True, alpha=0.2, facecolor=color)
                        plt.plot(-movement_metric[cluster][0,0:45], label = cluster, color = color)

        if feature == 'centroids':
            if mode == 'velocity':
                plt.yticks(np.arange(-10, 11, step=5), size=10)
                plt.ylim(-10, 10)
                plt.ylabel('Velocity', size = 12)
            elif mode == 'position':
                if animal in [2]:
                    plt.yticks(np.arange(-60, 91, step=30), size=10)
                    plt.ylim(-60, 90)
                elif animal in [3]:
                    if separate_traces:
                        plt.yticks(np.arange(-90, 91, step=30), size=10)
                        plt.ylim(-90, 90)
                    else:
                        plt.yticks(np.arange(-80, 81, step=40), size=10)
                        plt.ylim(-80, 80)
                elif animal in [4]:
                    plt.yticks(np.arange(-60, 121, step=60), size=10)
                    plt.ylim(-60, 120)
                else: # animal = 1
                    plt.yticks(np.arange(-120, 81, step=40), size=10)
                    plt.ylim(-120, 80)
                plt.ylabel('Distance', size = 12)
        elif feature == 'orientation':
            if mode == 'position':
                if separate_traces:
                    plt.ylim(-100, 100)
                    plt.yticks(np.arange(-80, 81, step=40), size=10)
                    plt.ylabel('Orientation (°)', size=12)
                else:
                    plt.ylim(-80, 80)
                    plt.yticks(np.arange(-60, 61, step=30), size=10)
                    plt.ylabel('Orientation (°)', size=12)
            elif mode == 'velocity':
                if separate_traces:
                    plt.yticks(np.arange(-8, 9, step=4), size=10)
                    plt.ylim(-8, 8)
                else:
                    plt.yticks(np.arange(-6, 7, step=3), size=10)
                    plt.ylim(-6, 6)
                plt.ylabel('Angular Velocity (°/frame)', size=12)

        plt.xlim(0, 30)
        plt.xticks(np.arange(0, 31, step=10), size  = 11)
        plt.minorticks_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.xaxis.set_minor_locator(mtp.ticker.MultipleLocator(5))
        plt.xlabel('Time (Frames)', size = 12)

        # create legend
        # if separate_traces:
        if create_Legend:
            LL = patches.Patch(color='blue', label='L, L')
            LR = patches.Patch(color='darkgreen', label='L, R')
            RR = patches.Patch(color='darkorange', label='R, R')
            RL = patches.Patch(color='red', label='R, L')
            Mixed = patches.Patch(color='yellow', label='Mixed')
            if mode == 'velocity':
                loc = 'upper right'
            else:
                loc = 'upper left'
            if len(list(movement_metric.keys())) > 4:
                plt.legend(frameon=False, handles=[LL, LR, RR, RL, Mixed],
                           fontsize = 7, loc = loc)
            else:
                plt.legend(frameon=False, handles=[LL, LR, RR, RL],
                           fontsize = 7, loc = loc)
        # else:
            # plt.legend(frameon=False, fontsize = 13)

        # SAVE PLOT
        if feature == 'centroids':
            if mode == 'velocity':
                if figure_title:
                    plt.title('LearnLight Velocity Plot (Clusters) Animal '+str(animal))
                if custom_filename is None:
                    plt.savefig('LearnLight_Velocity_Animal'+str(animal)+'.pdf', bbox_inches = 'tight')
                    plt.savefig('LearnLight_Velocity_Animal'+str(animal)+'.png', bbox_inches = 'tight',
                                                                                            dpi = 500)
                else:
                    plt.savefig(custom_filename+'LearnLight_Velocity_Animal'+str(animal)+'.pdf', bbox_inches = 'tight')
                    plt.savefig(custom_filename+'LearnLight_Velocity_Animal'+str(animal)+'.png', bbox_inches = 'tight',
                                                                                            dpi = 500)

            if mode == 'position':
                if figure_title:
                    plt.title('LearnLight Distance Plot (Clusters) Animal '+str(animal))
                if custom_filename is None:
                    plt.savefig('LearnLight_Distance_Animal'+str(animal)+'.pdf', bbox_inches = 'tight')
                    plt.savefig('LearnLight_Distance_Animal'+str(animal)+'.png', bbox_inches = 'tight',
                                dpi = 500)
                else:
                    plt.savefig(custom_filename+'LearnLight_Distance_Animal'+str(animal)+'.pdf', bbox_inches = 'tight')
                    plt.savefig(custom_filename+'LearnLight_Distance_Animal'+str(animal)+'.png', bbox_inches = 'tight',
                                dpi = 500)
        if feature == 'orientation':
            if mode == 'position':
                if figure_title:
                    plt.title('LearnLight Orientation Plot (Clusters) Animal '+str(animal))
                plt.savefig('LearnLight_Orientation_Animal'+str(animal)+'.pdf', bbox_inches = 'tight')
                plt.savefig('LearnLight_Orientation_Animal'+str(animal)+'.png', bbox_inches = 'tight', dpi = 500)
            if mode == 'velocity':
                if figure_title:
                    plt.title('LearnLight Angular Velocity Plot (Clusters) Animal '+str(animal))
                plt.savefig('LearnLight_AngularVelocity_Animal'+str(animal)+'.pdf', bbox_inches = 'tight')
                plt.savefig('LearnLight_AngularVelocity_Animal'+str(animal)+'.png', bbox_inches = 'tight', dpi = 500)
        plt.close('all')



################################# Mu Model Weight Analysis #########################

MuModelCoeff = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/Manuscript_Figures_(new)/MuModel/Weights/coef-2-new.npy', encoding = 'bytes')

## visualize weights and behavior trajectories

plt.ioff()

for animal in [1,2,3,4]:
    plots_per_sess = 9
    plot_trials = 90 // plots_per_sess
    behavior, _, _, trialtype, rewardinfo = combine_features(animal, all_sessions)
    nb_sessions = len(behavior)
    fig, ax = plt.subplots(nb_sessions, plots_per_sess,
                           figsize=(plots_per_sess * 2.3, nb_sessions*2))
    for sess in range(len(behavior)):
        for plot in range(plots_per_sess):
            start_idx = plot * plot_trials
            if plot == plots_per_sess:
                end_idx = -1
            else:
                end_idx = start_idx + plot_trials
            behav = behavior[sess][start_idx:end_idx,:]
            behav = np.reshape(behav[:, :200], (np.shape(behav)[0], 100, 2))
            dist = compute_euc_dist(behav)
            stim = trialtype[sess][start_idx:end_idx]
            x = np.arange(0, 40)  # convert to seconds
            y = np.transpose(dist[:, :40])
            if np.size(y[:, stim == 0]) != 0:
                ax[sess, plot].plot(x, y[:, stim == 0], c = 'red',
                                    linewidth = 1)
            if np.size(y[:, stim == 1]) != 0:
                ax[sess, plot].plot(x, y[:, stim == 1], c= 'blue',
                                    linewidth = 1)
            ax[sess, plot].set_xlim(0,30)
            ax[sess, plot].set_ylim(-110,110)
            ax[sess, plot].set_title('sess: '+str(sess+1) + ' group: ' +
                                     str(plot+1))
            ax[sess, plot].spines['right'].set_visible(False)
            ax[sess, plot].spines['top'].set_visible(False)
            ax[sess, plot].yaxis.set_ticks_position('left')
            ax[sess, plot].xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig('MuModelBehaviorComparison_anim'+str(animal)+'.png',
                bbox_inches='tight', pad_inches=0.1, dpi = 300)
    plt.close('all')

# weight plots only
for animalCount, MuModel_animal in enumerate(MuModelCoeff):
    nb_sessions = np.shape(MuModel_animal)[0]
    fig, ax = plt.subplots(nb_sessions, figsize=(17, nb_sessions*1.3))
    for sess in range(nb_sessions):
        session_WeightMatrix = np.transpose(MuModel_animal[sess])
        vis = ax[sess].matshow(session_WeightMatrix,
                         cmap=cm.get_cmap('RdBu'),
                               vmin=-6, vmax=6)
        ax[sess].set_axis_off()
    cb = fig.colorbar(vis, ax=ax.ravel().tolist())
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
    plt.savefig('MuModel_Weights_Visualize_Animal' +
                str(animalCount+1) + '.png', dpi = 300,
                bbox_inches='tight', pad_inches=0.003)
    plt.close()


##### PROCESS: histogram for Mu's enter-exit data
import pandas as pd

csv_dir = '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/Manuscript_Figures_(new)/Enter_Exit/MouseAcademy2_201711_corrected.csv'
df = pd.read_csv(csv_dir)

datetimeanimal = np.array(df[['Time_2', 'Time_3', 'Time_4', 'Time_5', 'Tag']])

timeanimal_splitday = [] # 1st dim: day. 2nd dim: row: enter, exit, animal
row_nb = 0
curr_date = np.array([0,0])
data_day = []
crossover = None
while row_nb < np.shape(datetimeanimal)[0] - 1:
    enter = datetimeanimal[row_nb]
    exit = datetimeanimal[row_nb + 1]
    date = enter[:2]
    enter_time = enter[2] + enter[3] / 60
    exit_time = exit[2] + exit[3] / 60
    animal = enter[-1]
    if np.array_equal(curr_date, date):
        if exit_time < enter_time: # crossed into next day
            crossover = [exit_time, animal]
            exit_time = 24
        data_day.append([enter_time, exit_time, animal])
    else:
        timeanimal_splitday.append(np.array(data_day))
        data_day = []
        if crossover is not None:
            # check if last animal crossed over into current day
            data_day.append([0, crossover[0], crossover[1]])
            crossover = None
        data_day.append([enter_time, exit_time, animal])
        curr_date = date
    row_nb += 2
timeanimal_splitday = timeanimal_splitday[1:]

# PLOT: duration plot for Mu's enter-exit data
fig, ax = plt.subplots(figsize = (6,5))
nb_days = len(timeanimal_splitday)
for day, data_day in enumerate(timeanimal_splitday):
    for entry in data_day:
        x = entry[:2]
        y = [day+1, day+1]
        animal = entry[-1]
        if animal == 1:
            color = 'blue'
        elif animal == 2:
            color = 'orange'
        elif animal == 3:
            color = 'green'
        elif animal == 4:
            color = 'red'
        ax.plot(x, y, c = color, linewidth = 3)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(0,24)
ax.set_ylim(0)
ax.set_ylabel('Day')
ax.set_xlabel('Hour')
plt.gca().invert_yaxis()  # invert axis
plt.savefig('Mice_EnterExit_Statistics.png', dpi = 200)
plt.close('all')

# PLOT: histogram of hours # convert to bins
animal_bins = [[],[],[],[]]
for animal in range(4):
    for data_day in timeanimal_splitday:
        data_animal_day = data_day[np.where(data_day[:,-1] == animal+1)]
        for entry in data_animal_day:
            start = entry[0]
            end = entry[1]
            binned = np.arange(start,end,1/60) # binned by minutes
            animal_bins[animal].extend(binned)

# plot histogram together
colors = ['blue', 'red', 'green', 'cyan']
plt.figure(figsize = (5,4.3))
for count in [3,4,1,2]:
    ani_histdata = animal_bins[count-1]
    plt.hist(ani_histdata, bins = 24,
             label = 'Animal '+str(count),
             histtype = 'stepfilled', alpha=0.75,
             edgecolor='black', linewidth=1,
             color = colors[count-1],
             density = True)
plt.xlabel('Hour of Day')
plt.ylabel('Probablity Density')
plt.xlim(0,24)
plt.ylim(0.01)
# sort legend
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]
plt.legend([handles[idx] for idx in order],
           [labels[idx] for idx in order],
           frameon=False)
plt.savefig('allanimalhist.png', dpi = 300)

# plot histogram separately
colors = ['blue', 'red', 'green', 'cyan']
for count, (ani_histdata, color) in enumerate(zip(animal_bins, colors)):
    plt.figure(figsize=(3.5, 3))
    plt.hist(ani_histdata, bins = 24,
             label = 'Animal '+str(count+1),
             histtype = 'stepfilled', alpha=1,
             color = color,
             edgecolor='black', linewidth=1,
             density = True)
    plt.xlabel('Hour of Day')
    plt.ylabel('Probability Density')
    plt.xlim(0,24)
    plt.ylim(0.01)
    # plt.legend(frameon = False)
    plt.tight_layout()
    plt.savefig(str(count+1)+'.png', dpi = 200)


# process csv files (REACTION TIME)

csv_folder = '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/Manuscript_Figures_(new)/MuModel/CSV_files'
def process_reactiontimes(csv_folder):
    file_dir = glob.glob(csv_folder + '/*.csv')
    file_dir.sort()

    all_animal_csv_data = []
    for file in file_dir:
        csvimport = np.loadtxt(file, delimiter=",", skiprows = 1)[:,[1,2,3,5,6]]
        nb_sessions = np.shape(csvimport)[0] / 90
        if (nb_sessions).is_integer():
            nb_trials = 90
            nb_features = np.shape(csvimport)[1]
            reshaped = np.reshape(csvimport, (int(nb_sessions),nb_trials,nb_features))
            all_animal_csv_data.append(reshaped)
        else:
            print('Error: session(s) with more/fewer than 90 trials found.')
    return all_animal_csv_data
all_animal_csv_data = process_reactiontimes(csv_folder)

## MATCH: reaction csv to actual tSNE trials

def match_csv_tSNE_trials():
    # this function computes pairwise alignment
    from Bio import pairwise2

    for animal in [1]:
        animal_rt = all_animal_csv_data[animal-1]
        animal_rt = np.reshape(animal_rt,
                               (np.shape(animal_rt)[0]*
                                np.shape(animal_rt)[1],
                                np.shape(animal_rt)[2]))
        trialinfo_csv = animal_rt[:,[0,2]]
        all_skipped = []
        # combine tSNE features
        combined_features, corresponding_sess, combined_startframes, \
        combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
        combined_trialtypes = np.reshape(combined_trialtypes, (np.size(combined_trialtypes),1))
        combined_rewardinfo = np.reshape(np.array(combined_rewardinfo)[:,-1], (np.shape(combined_rewardinfo)[0], 1))
        trialinfo_behavior = np.concatenate((combined_trialtypes, combined_rewardinfo), axis = 1)
        # prepare for alignment
        csv_align = trialinfo_csv[:,0].astype(int)
        csv_align_string = ''.join(map(str, csv_align))
        beh_align = trialinfo_behavior[:,0].astype(int)
        beh_align_string = ''.join(map(str, beh_align))
        # compute alignment
        alignments = pairwise2.align.globalxd(beh_align_string, csv_align_string,
                                              0, 0, -20, -20,
                                              one_alignment_only = True)
        deletion_idx = [pos for pos, char in enumerate(alignments[0][0]) if char == '-']

        animal_rt_matched = np.delete(animal_rt, deletion_idx, axis = 0)[:,-1]

        np.save('matched_reactiontimes_anim'+str(animal)+'.npy', animal_rt_matched)

        # Manual matching code (not functional)
        # while matched == False:
        #     trialinfo_csv_temp = np.delete(trialinfo_csv, all_skipped, axis = 0)
        #     output = trialinfo_csv_temp[0:np.shape(trialinfo_behavior)[0]] - trialinfo_behavior
        #     mismatch_idx = np.where(output != 0)[0]
        #     if np.size(mismatch_idx) > 0:
        #         mismatch_idx = mismatch_idx[0] # first occurence
        #         print('Mismatch detected..idx = ' + str(mismatch_idx + + len(all_skipped)))
        #         print(output[mismatch_idx:mismatch_idx + 20])
        #         skip_trials = input(
        #             'Input trials for skipping, separated by space: ')
        #         skip_list = [int(i) for i in skip_trials.split(' ') if i.isdigit()]
        #         all_skipped.extend(skip_list)
        #     else:
        #         print('match complete. saving animal '+str(animal))
        #         matched_animal_rt = np.delete(animal_rt, all_skipped, axis = 0)
        #         np.save('Matched_ReactionTimes_anim'+str(animal)+'.npy', matched_animal_rt)
        #         matched = True




# PLOT: weights + clustered states + reaction times
states = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/Manuscript_Figures_(new)/MuModel/Weights/clustered_states.npy')
for animalCount, MuModel_animal in enumerate(MuModelCoeff):
    nb_sessions = np.shape(MuModel_animal)[0]
    csvdata_animal = all_animal_csv_data[animalCount]
    fig, ax = plt.subplots(nb_sessions*3,
                           figsize=(17, nb_sessions*2),
                           )
    for sess in range(nb_sessions):
        session_WeightMatrix = np.transpose(MuModel_animal[sess])

        # PLOT: MUMODEL WEIGHT MATRICES
        vis = ax[sess * 3].matshow(session_WeightMatrix,
                                 cmap=cm.get_cmap('RdBu'),
                                 vmin=-4, vmax=4, aspect="auto"
                                 )
        ax[sess * 3].get_xaxis().set_ticks([])
        ax[sess * 3].get_yaxis().set_ticks([])
        # ax[sess*2].set_axis_off()
        # ax[sess*2].set_frame_on(True)

        # PLOT: STATES
        state_sess = np.reshape(states[animalCount][sess], (1,90))
        ax[sess * 3 + 1].matshow(state_sess, aspect="equal", cmap = 'prism')
        ax[sess * 3 + 1].get_xaxis().set_ticks([])
        ax[sess * 3 + 1].get_yaxis().set_ticks([])

        # PLOT: REACTION TIMES
        ax[sess * 3 + 2].plot(csvdata_animal[sess, :, -2],
                              label = 'Stim Duration', color = 'red')
        # share x-axis but use different y axis
        ax2 = ax[sess * 3 + 2].twinx()
        ax2.plot(csvdata_animal[sess, :, -1],
                 label = 'Reaction Time', color = 'blue')
        ax[sess * 3 + 2].set_xlim(0, 89)
        ax[sess * 3 + 2].set_ylim(0, 9)
        ax[sess * 3 + 2].tick_params('y', colors='red')
        ax[sess * 3 + 1].get_xaxis().set_ticks([])
        ax2.get_xaxis().set_ticks([])
        ax2.tick_params('y', colors='blue')
        ax2.set_ylim(0,2)
        # ax[sess * 2 + 1].set_axis_off()
        # ax[sess * 3 + 1].get_xaxis().set_visible(False)
        if sess == 0:
            ax[sess * 3 + 2].legend(frameon = False, loc = 'center left')
            ax2.legend(frameon=False, loc='center right')
    # cb = fig.colorbar(vis, ax=ax.ravel().tolist())
    # cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
    plt.savefig('MuModel_Weights_Visualize_Animal' +
                str(animalCount+1) + '.png', dpi = 300,
                bbox_inches='tight', pad_inches=0.003)
    plt.close()

##################### correlation analysis ###############################
# correlation between weights and animal distances (direction normalized)

for animal in [1,2,3,4]:
    behavior, _, _, trialtype, rewardinfo = combine_features(animal, all_sessions)
    weights = MuModelCoeff[animal-1]
    combined_X = []
    for sess in range(len(behavior)):
        behav = behavior[sess]
        behav = np.reshape(behav[:, :200], (np.shape(behav)[0], 100, 2))
        center_port = np.average(behav[:,0,:], axis=0)
        center_port[1] = 325 # set y value to center of video
        dist = compute_euc_dist(behav, centerport = center_port,
                                normalize=False)
        # dist = dist / 80 # normalize
        weights_sess = weights[sess]
        for trial_nb, weights_trial in enumerate(weights_sess):
            if all(weights_trial == 0):
                dist = np.insert(dist, trial_nb, np.zeros((100)), axis = 0)
                behavior[sess] = np.insert(behavior[sess], trial_nb, np.zeros(300), axis = 0)
                ## add zero padding for missing trial
            ### combine into single array for correlation analysis
            if trial_nb < len(dist):
                # COMBINE WEIGHTS AND FEATURES INTO ONE ARRAY COMBINED_X
                # combined_X.append(np.concatenate((weights_trial,dist[trial_nb][5:30:5])))
                # combine weights and surrogate measures (start position, orientation)
                start_dist = dist[trial_nb][0]
                start_orient = behavior[sess][trial_nb, 200]
                # time it takes animal to move
                # latency =
                combined_X.append(np.concatenate((weights_trial,[start_dist],[start_orient])))

    combined_X = np.transpose(np.array(combined_X))
    # compute pearson's correlations
    nb_weights = np.shape(weights)[-1]
    behavior_dim = np.shape(combined_X)[0] - nb_weights
    corrmatrix = np.zeros((nb_weights, behavior_dim))
    for weight_idx in range(nb_weights):
        for dist_idx in range(behavior_dim):
            x = combined_X[weight_idx]
            y = combined_X[weight_idx + dist_idx+1]
            correlationP = pearsonr(x,y)[0]
            corrmatrix[weight_idx, dist_idx] = correlationP
    # PLOT
    plt.figure()
    plt.matshow(corrmatrix, cmap = 'RdBu')
    # x = [0,1,2,3,4]
    # xlabels = [0.17, 0.33, 0.5, 0.67, 0.83]
    x = [0,1]
    xlabels = ['Start distance', 'Start orientation']
    plt.gca().xaxis.tick_bottom()
    plt.xticks(x, xlabels)
    plt.xlabel('Behavior Features')
    y = [0,1,2,3,4,5,6,7]
    ylabels = ['Stim',
              'Choice/Reward Interaction (-3)',
              'Choice/Reward Interaction (-2)',
              'Choice/Reward Interaction (-1)',
              'Choice history (-3)',
              'Choice history (-2)',
              'Choice history (-1)',
              'Bias']
    plt.yticks(y, ylabels,  rotation='horizontal')
    plt.ylabel('Model Weights')
    plt.tight_layout()
    plt.colorbar()
    plt.title('Correlation Matrix for Animal '+str(animal))
    plt.savefig('CorrWeightBehav_Anim'+str(animal)+'.png',dpi=200,bbox_inches='tight')
    plt.close('all')




    #     if np.size(y[:, stim == 0]) != 0:
    #         ax[sess, plot].plot(x, y[:, stim == 0], c = 'red',
    #                             linewidth = 1)
    #     if np.size(y[:, stim == 1]) != 0:
    #         ax[sess, plot].plot(x, y[:, stim == 1], c= 'blue',
    #                             linewidth = 1)
    #     ax[sess, plot].set_xlim(0,30)
    #     ax[sess, plot].set_ylim(-110,110)
    #     ax[sess, plot].set_title('sess: '+str(sess+1) + ' group: ' +
    #                              str(plot+1))
    #     ax[sess, plot].spines['right'].set_visible(False)
    #     ax[sess, plot].spines['top'].set_visible(False)
    #     ax[sess, plot].yaxis.set_ticks_position('left')
    #     ax[sess, plot].xaxis.set_ticks_position('bottom')
    # plt.tight_layout()
    # plt.savefig('MuModelBehaviorComparison_anim'+str(animal)+'.png',
    #             bbox_inches='tight', pad_inches=0.1, dpi = 300)
    # plt.close('all')



## cluster weights into categories

allWeightVectors = []

for MuModel_animal in MuModelCoeff:
    for sess in MuModel_animal:
        for trial in sess:
            allWeightVectors.append(trial)

allWeightVectors = np.array(allWeightVectors)



for perplexity in [5,17,35,50,70,100]:
    PCArun = PCA(n_components=2, whiten=False)
    embedded = PCArun.fit_transform(allWeightVectors)
    # plot embedding
    plt.figure()
    plt.scatter(embedded[:,0], embedded[:,1], edgecolor='black',
                linewidths=0.3, s = 12, color = 'red')
    plt.savefig('weight_embedding_PCA',dpi = 300)

#####################################3
## debugging reward file match errors
# animal = 4
# session = 'Animal4_LearnLight2AFC_20171105T012921.mj2'
# Flash = np.load(flash_detection_folder+'/'+session+'_LED_and_nosepoke_flashes.npy')[()]
# nosepoke_flashes_raw = Flash['nosepoke_flashes']
# trialtype_raw = Flash['trial_types']
# trialstart_frames, trialtypes_j_video = process_raw_flashes(nosepoke_flashes_raw, trialtype_raw)
# all_sessions = list(Data_for_tSNE.keys())
# all_sess_1animal = [sess for sess in all_sessions if
#                     sess[6] == str(animal)]  # names of sessions for current animal number
# all_sess_1animal = [s for s in all_sess_1animal if len(all_trialtypes[s]) > 80]
#
# i = files[animal-1]
# csvimport = np.loadtxt(i, delimiter=",", skiprows=1)[:, :4]
# _, sessionidxs = np.unique(csvimport[:, 0], return_index=True)
# sessionidxs = np.append(sessionidxs, csvimport.shape[0])  # add idx of last trial
#
#
# assert len(all_sess_1animal) == len(sessionidxs) - 1, 'Error: session length mismatch. File: ' + str(i)
#
# j = all_sess_1animal.index(session+'_pose.json')
# startidx = sessionidxs[j]
# endidx = sessionidxs[j + 1]
# trialtypes_j = csvimport[startidx:endidx, 1]
#
# print(np.concatenate((np.indices((len(trialtypes_j_video), 1))[0],
#                       np.reshape(trialtypes_j_video, (len(trialtypes_j_video), 1)),
#                       np.reshape(trialtypes_j[0:len(trialtypes_j_video)], (len(trialtypes_j_video), 1))), 1))

## run this to process data
# process_data()


# all_sessions = list(Data_for_tSNE.keys())

### PLOT: tSNE clusters based on manual clustering + reaction times for animals 3 and 4

def tSNE_cluster_plot_reactiontime():
    visualize_Box = False
    for animal in [1,2,3,4]:
        embedding = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                            'tSNE_learnlight_animal' + str(animal) +
                            '_window135/embedded_animal' + str(animal) + '.npy')
        if animal == 1:
            clusters_coord = {
                # 'CLUSTER': [x1, x2, y1, y2]
                'LL': np.array([[-70, -37, -40, 3],
                                [-70, 6, 3, 60],
                                [-37, -32, -8, 8]]),
                'RR': np.array([[45, 80, -20, 40],
                                [6, 45, 25, 50],
                                [33, 45, 19, 25]]),
                'LR': np.array([[10, 45, -10, 18],
                                [7, 10, 0, 10]]),
                'RL': np.array([[-35, 10, -50, 0],
                                [-20, 5, 0, 10],
                                [-39, -35, -50, -25]])
            }
        elif animal == 2:
            clusters_coord = {
                # 'CLUSTER': [x1, x2, y1, y2]
                # 'LL': np.array([[17, 40, -30, 20],
                #                 [5, 18, -30, -5]]),
                'LL': np.array([[17, 40, -30, 20],
                                ]),
                'RR': np.array([[-40, -20, -10, 40]]),
                'LR': np.array([[-20, 4, -20, -5], [-20, -8, -5, 0.5]]),
                'RL': np.array([[5, 17, -5, 12]]),
                'Unknown': np.array([[5, 18, -30, -6]]),
                'Mixed': np.array([[-7, 4, -5, 10],[-20, 4, 0.5, 10]])
            }
        elif animal == 3:
            clusters_coord = {
                # 'CLUSTER': [x1, x2, y1, y2]
                'LL': np.array([[-60, -30, -60, -20],
                                [-30, 24, -80, -37],
                                [-30, -25, -37, -33],
                                [24, 30, -80, -50]]),
                'RR': np.array([[-40, 20, 40, 80], [20, 50, 8, 60]]),
                'LR': np.array([[-50, 10, -5, 40]]),
                'RL': np.array([[20, 60, -45, 0]]),
                'Mixed': np.array([[-17, 11, -37, -2],
                                   [-24, -17, -36, -20]])
            }
        elif animal == 4:
            clusters_coord = {
                # 'CLUSTER': [x1, x2, y1, y2]
                'LL': np.array([[-70, 15, -40, 0], [-70, -31, 0, 40], [-70, -20, 25, 40]]),
                'RR': np.array([[-4, 70, 5, 50], [40, 79, -40, 5]]),
                'LR': np.array([[0, 30, -70, -30]]),
                'RL': np.array([[-35, -5, 0, 12],
                                [-30, -5, 11, 28],
                                [-21, -16, 28, 30.5]]),
                'Mixed': np.array([[-8, 30, -20, 5]])
            }
        # extract idxs
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        for cluster_type in list(clusters_coord.keys()):
            cluster = clusters_coord[cluster_type]
            idxs = np.array([], dtype='int')
            embedding_x = embedding[:,0]
            embedding_y = embedding[:,1]
            for block in cluster:
                x1, x2, y1, y2 = block[0], block[1], block[2], block[3]
                ## tally up all points
                idxs = np.append(idxs, np.where((embedding_x >= x1) & (embedding_x <= x2) &
                                                (embedding_y >= y1) & (embedding_y <= y2))[0])
                idxs = np.unique(idxs)
            ### plot, compute reaction info for this cluster
            if cluster_type == 'Mixed':
                ax.scatter(embedding[idxs, 0], embedding[idxs, 1],
                           s=10, edgecolor='black', linewidths=0.3, label = cluster_type,
                           color = 'cyan')
            else:
                ax.scatter(embedding[idxs, 0], embedding[idxs, 1],
                           s=10, edgecolor='black', linewidths=0.3, label = cluster_type)
        # adjust axis number size
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        ## plot box around clusters to ensure full coverage
        if visualize_Box:
            for cluster_type in list(clusters_coord.keys()):
                cluster = clusters_coord[cluster_type]
                for block in cluster:
                    min_X = min(block[:2])
                    min_Y = min(block[2:])
                    width = max(block[:2]) - min_X
                    height = max(block[2:]) - min_Y
                    polygon = plt.Rectangle([min_X, min_Y], width, height, fill = None,
                                          edgecolor = 'red', linewidth=0.5,
                                          )
                    plt.gca().add_patch(polygon)
        # other plot configs
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
            # ax.set_xticks([])
            # ax.set_yticks([])
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        # plt.legend(frameon = False)
        plt.savefig('tSNE_clusters_anim'+str(animal)+'.png', bbox_inches='tight', dpi = 500)
        # plt.savefig('tSNE_clusters_anim'+str(animal)+'.pdf', bbox_inches='tight')
        # plt.savefig('tSNE_clusters_legend_anim'+str(animal)+'.png', bbox_inches='tight', dpi = 500)
        plt.close('all')

        ### plot reaction plots:
        anim_rt = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/'
                          'Mouse_tracking_analysis/Manuscript_Figures_(new)/'
                          'matched_reactiontimes_anim'+str(animal)+'.npy')

        fig, ax = plt.subplots(figsize=(3.5, 3))
        cluster_idxs = {}
        all_data = []
        for nb, cluster_type in enumerate(list(clusters_coord.keys())):
            cluster = clusters_coord[cluster_type]
            idxs = np.array([], dtype='int')
            embedding_x = embedding[:,0]
            embedding_y = embedding[:,1]
            for block in cluster:
                x1, x2, y1, y2 = block[0], block[1], block[2], block[3]
                ## tally up all points
                idxs = np.append(idxs, np.where((embedding_x >= x1) & (embedding_x <= x2) &
                                                (embedding_y >= y1) & (embedding_y <= y2))[0])
                idxs = np.unique(idxs)
            cluster_idxs[str(nb+1)] = idxs
            data = anim_rt[idxs]
            # remove timed out trials (data == 10 seconds)
            timeout_idxs = np.where(data == 10)[0]
            data = np.delete(data, timeout_idxs)
            all_data.extend(data)
            if cluster_type == 'Mixed':
                parts = ax.violinplot(data, [nb], showmeans = True, widths = 0.6)
                # now change colors
                for pc in parts['bodies']:
                    pc.set_facecolor('cyan')
                    pc.set_alpha(1)
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1.5)
            else:
                ax.violinplot(data, [nb], showmeans = True, widths = 0.6)
        # plot 6th cluster with all data points
        parts = ax.violinplot(all_data, [nb+1], showmeans = True, widths = 0.6)
        # now change colors
        for pc in parts['bodies']:
            pc.set_facecolor('C7')
            pc.set_alpha(0.3)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = parts[partname]
            vp.set_edgecolor('C7')
            vp.set_linewidth(1.5)
        #######################
        # ax.set_yscale('log')
        ax.set_ylim(0)
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Reaction Time (s)')
        if animal in [3,4]:
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            xlabel = ['1','2','3','4','5','All']
        elif animal in [2]:
            ax.set_xticks([0,1,2,3,4,5,6])
            xlabel = ['1', '2', '3', '4', '5', '6', 'All']
        else:
            ax.set_xticks([0, 1, 2, 3, 4])
            xlabel = ['1','2','3','4','All']
        ax.set_xticklabels(xlabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        plt.savefig('ReactionTimes_tSNE_clusters'+str(animal)+'.png',
                    bbox_inches='tight', dpi = 300)

        ## behavior traces
        plot_dist_to_nosepoke(animal, cluster_idxs = cluster_idxs,
                              embedding = embedding, separate_traces = False,
                              mode = 'position', feature = 'centroids',
                              create_Legend = False)
        plot_dist_to_nosepoke(animal, cluster_idxs = cluster_idxs,
                              embedding = embedding, separate_traces = False,
                              mode = 'velocity', feature = 'centroids',
                              create_Legend=False)
        plot_dist_to_nosepoke(animal, cluster_idxs = cluster_idxs,
                              embedding = embedding, separate_traces = False,
                              mode = 'position', feature = 'orientation',
                              create_Legend=False)
        plot_dist_to_nosepoke(animal, cluster_idxs = cluster_idxs,
                              embedding = embedding, separate_traces = False,
                              mode = 'velocity', feature = 'orientation',
                              create_Legend=False)


################## SPATIAL PLOT  #############################################

for animal in [1,2,3,4]:
    combined_features, corresponding_sess, \
    combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    combined_trialtypes = np.array(combined_trialtypes)
    combined_rewardinfo = np.array(combined_rewardinfo)[:,-1]
    nb_frames = 40
    centroids = np.array(combined_features)[:, 0:2 * nb_frames]
    centroidsX = centroids[:,0::2]
    # centroidsX = np.transpose(centroidsX)
    centroidsY = centroids[:,1::2]
    # centroidsY = np.transpose(centroidsY)
    plt.figure(figsize = (1.5,3))
    plt.xlim(500,380)
    # zoomed in
    # plt.ylim(460,175)
    # actual video size
    plt.ylim(650,0)
    plt.xticks([])
    plt.yticks([])
    ax = plt.axes()
    ax.set_aspect(aspect = 'equal')
    for centroid_X, centroid_Y, stim in zip(centroidsX, centroidsY, combined_trialtypes):
        if stim == 0:
            color = 'C0'
        else:
            color = 'C1'
        plt.plot(centroid_X, centroid_Y, color = color,
                 linewidth = 0.2, alpha = 0.15, zorder = 1)
    for count, (centroid_X, centroid_Y, stim) in enumerate(zip(centroidsX, centroidsY, combined_trialtypes)):
        if stim == 0:
            color = 'C0'
        else:
            color = 'C1'
        # indicate starting position with marker
        plt.scatter(centroid_X[0], centroid_Y[0], s = 0.15, c = color,
                    edgecolor = 'black', linewidths = 0.05, zorder = 2, marker = 'o')
        # plt.annotate(count, (centroid_X[0], centroid_Y[0]), fontsize=0.2)

    plt.savefig(str(animal)+'_spatialplot.png', dpi = 800, bbox_inches = 'tight')
    plt.savefig(str(animal)+'_spatialplot.pdf', bbox_inches = 'tight')
    plt.close('all')

##################### LDA #####################################
##################### NEW PCA PLOTTING!!! #####################
mode = 'PCA' #### PCA or LDA
separate_axis = False
for animal in [1,2,3,4]: # iterate over animals
    combined_features, corresponding_sess, \
    combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    combined_trialtypes = np.array(combined_trialtypes)
    combined_rewardinfo = np.array(combined_rewardinfo)[:,-1]
    for nb_frames in [30]: # iterate over the number of frames to feed into the algorithm
        centroids = np.array(combined_features)[:, 0:2*nb_frames]
        orientations = np.array(combined_features)[:, 200:200+nb_frames]
        truncated_combined_features = np.concatenate((centroids,orientations), axis = 1)
        ## add reaction time?
        rt_all = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/'
                     'Mouse_tracking_analysis/Manuscript_Figures_(new)/'
                     'matched_reactiontimes_anim'+str(animal)+'.npy')
        # truncated_combined_features = np.concatenate((truncated_combined_features, rt), axis = 1)
        # compute class labels
        Y = []
        colors = []
        behavior = []
        reward = []
        # create separate min and max for coloring reaction times
        rt_minmax_grouped = {}
        ## LL
        idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 0),
                              np.where(np.array(combined_rewardinfo) == 1))
        rt_minmax_grouped['LL'] = (np.log10(np.min(rt_all[idxs])),
                                   np.log10(np.max(rt_all[idxs])))
        # LR
        idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 0),
                              np.where(np.array(combined_rewardinfo) == -1))
        rt_minmax_grouped['LR'] = (np.log10(np.min(rt_all[idxs])),
                                   np.log10(np.max(rt_all[idxs])))
        # RR
        idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 1),
                              np.where(np.array(combined_rewardinfo) == 1))
        rt_minmax_grouped['RR'] = (np.log10(np.min(rt_all[idxs])),
                                   np.log10(np.max(rt_all[idxs])))
        # RL
        idxs = np.intersect1d(np.where(np.array(combined_trialtypes) == 1),
                              np.where(np.array(combined_rewardinfo) == -1))
        rt_minmax_grouped['RL'] = (np.log10(np.min(rt_all[idxs])),
                                   np.log10(np.max(rt_all[idxs])))

        ## plot starting position as hue
        all_start_yaxis = truncated_combined_features[:,1]
        all_start_yaxis_norm = (all_start_yaxis - np.min(all_start_yaxis)) / \
                               np.max((all_start_yaxis - np.min(all_start_yaxis)))
        all_start_yaxis_norm = 0.7 * all_start_yaxis_norm + 0.2
        for trial, outcome, rt, start_y in zip(combined_trialtypes, combined_rewardinfo, rt_all, all_start_yaxis_norm):
            # NORMALIZE ACROSS ALL GROUPS
            # min_rt = np.log10(np.min(rt_all))
            log_rt = np.log10(rt)
            max_rt = -0.5
            min_color = 0.4
            max_color = 0.8
            # rt_norm = min_color + (max_color - min_color) * ((log_rt - min_rt) / (max_rt - min_rt))
            ## LL
            if trial == 0 and outcome == 1:
                Y.append(0)
                # colors.append('blue')
                # min_rt, max_rt = rt_minmax_grouped['LL']
                # rt_norm = min_color + (max_color - min_color) * ((log_rt - min_rt) / (max_rt - min_rt))
                # colors.append(cm.Blues(rt_norm))
                colors.append(cm.Blues(start_y))
                behavior.append(0)
                reward.append(1)
            ## RR
            elif trial == 1 and outcome == 1:
                Y.append(1)
                # colors.append('darkorange')
                # min_rt, max_rt = rt_minmax_grouped['RR']
                # rt_norm = min_color + (max_color - min_color) * ((log_rt - min_rt) / (max_rt - min_rt))
                # colors.append(cm.Oranges(rt_norm))
                colors.append(cm.Oranges(start_y))
                behavior.append(1)
                reward.append(1)
            # LR
            elif trial == 0 and (outcome == -1 or outcome == 0):
                Y.append(2)
                # colors.append('darkgreen')
                # min_rt, max_rt = rt_minmax_grouped['LR']
                # rt_norm = min_color + (max_color - min_color) * ((log_rt - min_rt) / (max_rt - min_rt))
                # colors.append(cm.Greens(rt_norm))
                colors.append(cm.Greens(start_y))
                behavior.append(1)
                reward.append(0)
            # RL
            elif trial == 1 and (outcome == -1 or outcome == 0):
                Y.append(3)
                # colors.append('red')
                # min_rt, max_rt = rt_minmax_grouped['RL']
                # rt_norm = min_color + (max_color - min_color) * ((log_rt - min_rt) / (max_rt - min_rt))
                # colors.append(cm.Reds(rt_norm))
                colors.append(cm.Reds(start_y))
                behavior.append(0)
                reward.append(0)
        ## COMPUTE LDA
        if mode == 'LDA':
            if separate_axis:
                x_clf = LDA(n_components = 1, solver = 'svd')
                y_clf = LDA(n_components = 1, solver = 'svd')
            else:
                clf = LDA(n_components = 3, solver = 'svd')
        elif mode == 'PCA':
            clf = PCA(n_components = 3, whiten = False)
        if separate_axis: # this code is for LDA only
            ####
            # currently use only y position as features
            y_pos = truncated_combined_features[:,1:60:2]
            X = x_clf.fit_transform(y_pos, combined_trialtypes)
            Y = y_clf.fit_transform(y_pos, behavior)
            # plot here
            plt.figure(figsize = (3.7,3.7))
            plt.scatter(X, Y, c = colors, s = 15,
                        edgecolor='black', linewidths=0.4)
            plt.xlabel(mode+' 1')
            plt.ylabel(mode+' 2')
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ### LEGEND! code copied from distance plots
            LL = patches.Patch(color='blue', label='L, L')
            LR = patches.Patch(color='darkgreen', label='L, R')
            RR = patches.Patch(color='darkorange', label='R, R')
            RL = patches.Patch(color='red', label='R, L')
            Mixed = patches.Patch(color='yellow', label='Mixed')
            loc = ['upper left',
                   'upper left',
                   'upper left',
                   'upper right']
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(frameon=False, handles=[LL, LR, RR, RL],
                       fontsize = 7, loc = loc[animal-1])
            # save figures
            plt.savefig(str(animal)+'_'+str(nb_frames)+'_'+mode+'_sepaxis.png', dpi = 500,
                        bbox_inches='tight')
            plt.close('all')

            # PLOT SCALING
            x_scaling = x_clf.scalings_
            y_scaling = y_clf.scalings_
            print(x_clf.explained_variance_ratio_)
            print(y_clf.explained_variance_ratio_)
            # x_positions = np.reshape(x_scaling[0:2 * nb_frames, :], (1, nb_frames, 2))
            # y_positions = np.reshape(y_scaling[0:2 * nb_frames, :], (1, nb_frames, 2))
            # PLOT DISTANCE
            # x_dists = compute_euc_dist(x_positions, centerport=None)
            # y_dists = compute_euc_dist(y_positions, centerport=None)
            plt.figure(figsize=(3.8, 3.5))
            # plt.plot(x_dists[0], label='Trial')
            # plt.plot(y_dists[0], label='Behavior')
            # plt.plot(x_positions[0,:,1], label='Trial')
            # plt.plot(y_positions[0,:,1], label='Behavior')
            plt.plot(x_scaling, label='Trial')
            plt.plot(y_scaling, label='Behavior')
            plt.legend(frameon=False)
            plt.xlabel('Time (Frames)')
            plt.xticks(np.arange(0, nb_frames + 1, step=10), size=11)
            plt.xlim(0, nb_frames)
            plt.savefig(str(animal) + mode + '_scalingdist_sepaxis.png', bbox_inches='tight',
                        dpi=500)

        else:
            trans = clf.fit_transform(truncated_combined_features, Y)
            var = clf.explained_variance_ratio_
            print(var)
            # np.save('LDA_embedding30_anim_rt'+str(animal)+'.npy', trans)
            ############# Scaling analysis
            if mode == 'LDA':
                scaling = clf.scalings_
                positions = np.reshape(scaling[0:2 * nb_frames, :], (nb_frames, 2, 3))
                orientations = scaling[2 * nb_frames:,:]
                # plot scaling directly
                plt.figure(figsize=(4.5, 4))
                for dim in range(3):
                    traj = positions[:,:,dim]
                    label = str(dim + 1) + ' expvar = ' + str(var[dim])
                    plt.plot(traj[:, 0], traj[:, 1], label=label, marker='x')
                plt.legend(frameon=False)
                # plt.ylim(-0.2, 0.35)
                plt.savefig(str(animal)+'LDA_scaling_3d.png', bbox_inches = 'tight',
                            dpi = 500)
                # PLOT DISTANCE
                positions = np.transpose(positions, (2,0,1))
                dists = compute_euc_dist(positions, centerport=None)
                plt.figure(figsize=(3.8, 3.5))
                for count, dist in enumerate(dists):
                    label = 'Dim ' + str(count + 1) + ' expvar = ' \
                            + str(var[count])
                    # plt.plot(abs(dist), label=label)
                    plt.plot(dist, label=label)
                plt.legend(frameon=False)
                plt.xlabel('Time (Frames)')
                plt.xticks(np.arange(0, nb_frames + 1, step=10), size=11)
                plt.xlim(0, nb_frames)
                plt.savefig(str(animal)+mode+'_scalingdist_3d_abs.png', bbox_inches = 'tight',
                            dpi = 500)
            #### OUTLIER REMOVE ANIMAL 1
            if animal == 1:
                # remove outlier
                outlier_idx = np.where(trans[:,2] > 200)[0]
                trans = np.delete(trans, outlier_idx, 0)
                colors = np.delete(colors, outlier_idx, 0)
            ################ 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax = fig.add_subplot(projection = '3d')

            ax.scatter(trans[:,0], trans[:,1], trans[:,2],
                       c = colors, s = 10, edgecolor = 'black',
                       linewidths = 0.4)
            ax.set_xlabel(mode+' 1')
            ax.set_ylabel(mode+' 2')
            ax.set_zlabel(mode+' 3')
            ax.view_init(30, 130)
            ax.dist = 12
            plt.tick_params(labelsize=7.5)
            # ### LEGEND! code copied from distance plots
            # LL = patches.Patch(color='blue', label='L, L')
            # LR = patches.Patch(color='darkgreen', label='L, R')
            # RR = patches.Patch(color='darkorange', label='R, R')
            # RL = patches.Patch(color='red', label='R, L')
            # Mixed = patches.Patch(color='yellow', label='Mixed')
            # loc = 'upper left'
            # plt.legend(frameon=False, handles=[LL, LR, RR, RL],
            #            fontsize = 7, loc = loc)
            plt.savefig(str(animal)+'_'+str(nb_frames)+'_3D_'+mode+'.png',
                        dpi = 500, bbox_inches = 'tight')
            # 2D plot
            plt.figure(figsize = (3.7,3.7))
            #### PCA 1+3
            plt.scatter(trans[:, 0], trans[:, 2], c=colors, s=15,
                        edgecolor='black', linewidths=0.4)
            plt.ylabel(mode+' 3', fontsize = 12)
            #### PCA 1+2
            # plt.scatter(trans[:,0], trans[:,1], c = colors, s = 15,
            #             edgecolor='black', linewidths=0.4)
            # plt.ylabel(mode+' 2', fontsize = 12)
            ##
            plt.xlabel(mode + ' 1', fontsize = 12)
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # adjust axis number size
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            ### LEGEND! code copied from distance plots
            LL = patches.Patch(color='blue', label='L, L')
            LR = patches.Patch(color='darkgreen', label='L, R')
            RR = patches.Patch(color='darkorange', label='R, R')
            RL = patches.Patch(color='red', label='R, L')
            Mixed = patches.Patch(color='yellow', label='Mixed')
            loc = 'upper left'
            plt.legend(frameon=False, handles=[LL, LR, RR, RL],
                       fontsize = 7, loc = loc)
            # save figures
            plt.savefig(str(animal)+'_'+str(nb_frames)+'_'+mode+'.png', dpi = 500,
                        bbox_inches='tight')
            plt.close('all')

            ############ plot behavior traces
            ### split the 2nd/3rd axis into 4 chunks
            window = 4
            for axis in [2,3]:
                embedding = trans[:,(0,axis-1)]
                min, max = np.min(embedding[:,-1]), np.max(embedding[:,-1])
                window_size = int((max - min) // 4)
                for wind in range(window):
                    wind_range = (min + wind * window_size,
                                  min + (wind + 1) * window_size)
                    # compute points inside window
                    y1, y2 = wind_range[0], wind_range[1]
                    wind_data_idxs = np.where((embedding[:,-1] >= y1) & (embedding[:,-1] <= y2))[0]
                    # create cluster_idxs dictionary object
                    cluster_idxs = {}
                    ## LL
                    idxs = np.intersect1d(np.where(combined_trialtypes == 0),
                                          np.where(combined_rewardinfo == 1))
                    intersect_idxs = np.intersect1d(idxs, wind_data_idxs)
                    cluster_idxs['LL'] = intersect_idxs
                    ## RR
                    idxs = np.intersect1d(np.where(combined_trialtypes == 1),
                                          np.where(combined_rewardinfo == 1))
                    intersect_idxs = np.intersect1d(idxs, wind_data_idxs)
                    cluster_idxs['RR'] = intersect_idxs
                    ## LR
                    idxs = np.intersect1d(np.where(combined_trialtypes == 0),
                                          np.union1d(np.where(combined_rewardinfo == -1)[0],
                                                     np.where(combined_rewardinfo == 0)[0]))
                    intersect_idxs = np.intersect1d(idxs, wind_data_idxs)
                    cluster_idxs['LR'] = intersect_idxs
                    ## RL
                    idxs = np.intersect1d(np.where(combined_trialtypes == 1),
                                          np.union1d(np.where(combined_rewardinfo == -1)[0],
                                                     np.where(combined_rewardinfo == 0)[0]))
                    intersect_idxs = np.intersect1d(idxs, wind_data_idxs)
                    cluster_idxs['RL'] = intersect_idxs
                    #### plot behavior traces
                    plot_dist_to_nosepoke(animal, cluster_idxs = cluster_idxs,
                                          custom_filename = str(animal)+'_axis'+str(axis)+'_wind'+str(wind))

####################################
plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

    ### create folder where all the output files will be saved

####################### tSNE plot scripts #######################

for animal in [1,2,3,4]: # iterate over animals
    combined_features, corresponding_sess, \
    combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    for nb_frames in [45]: # iterate over the number of frames to feed into the algorithm
        # truncated_combined_features = np.array(combined_features)[:,:j]
        centroids = np.array(combined_features)[:, 0:2*nb_frames]
        orientations = np.array(combined_features)[:, 200:200+nb_frames]
        truncated_combined_features = np.concatenate((centroids,orientations), axis = 1)
        compute_and_plot_tSNE(truncated_combined_features, combined_startframes, combined_trialtypes,
                              combined_rewardinfo, corresponding_sess, label = False, save = True,
                              plot_per_session = False, plot_cluster_centroids = True)
        plt.close('all')

    ############### TEST
    # animal = 3
    # combined_features, corresponding_sess, \
    # combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    # compute_and_plot_tSNE(np.array(combined_features)[:,:91], combined_startframes, combined_trialtypes,
    #                       combined_rewardinfo, corresponding_sess, label = False, save = True,
    #                       plot_per_session = False, recompute_tSNE = True, plot_cluster_centroids = False)

################### UNSUPERVISED kMEANS CLUSTERING ON tSNE/PCA FEATURES
from scipy.cluster.vq import whiten
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import homogeneity_completeness_v_measure as compute_HCV
import itertools


def compute_and_plot_HCV(labels, combined_trialtypes, combined_rewardinfo):
    label_iter = list(itertools.permutations([0, 1, 2, 3]))
    HCV = []
    for iter in label_iter:
        trueLabels = []
        rewardinfo = np.array(combined_rewardinfo)[:,-1]
        labelmap = {(0, 1): iter[0],
                    (0, -1): iter[1],
                    (0, 0): iter[1],
                    (1, 1): iter[2],
                    (1, -1): iter[3],
                    (1, 0): iter[3]}
        for stim, reward in zip(combined_trialtypes, rewardinfo):
            trueLabels.append(labelmap[stim,reward])
        HCV.append(compute_HCV(trueLabels, labels))
    HCV = np.array(HCV)
    idx = np.argmin(HCV, axis = 0)[-1]
    print('Lowest v: '+str(idx))
    pdb.set_trace()
    return HCV[idx]


embeddings = ['/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/tSNE_learnlight_animal1_window135/embedded_animal1.npy',
              '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/tSNE_learnlight_animal2_window135/embedded_animal2.npy',
              '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/tSNE_learnlight_animal3_window135/embedded_animal3.npy',
              '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/tSNE_learnlight_animal4_window135/embedded_animal4.npy']

HCV_all = []
for animal, clusters, embedding_dir in zip([1,2,3,4], [4,4,5,5], embeddings):
    plt.ioff()
    embedded = np.load(embedding_dir)
    nb_frames = 45
    combined_features, corresponding_sess, combined_startframes, \
    combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    # truncate features
    centroids = np.array(combined_features)[:, 0:2 * nb_frames]
    orientations = np.array(combined_features)[:, 200:200 + nb_frames]
    truncatedFeatures = np.concatenate((centroids, orientations), axis=1)
    # whiten feature
    # whitenedFeatures = whiten(truncatedFeatures)
    # fit kmeans
    # kmeans = KMeans(n_clusters = clusters, init = 'random').fit(truncatedFeatures)
    # labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    # fit GMM

    for clusters in range(4,5):
        print('* Animal '+str(animal))
        print('  Cluster '+str(clusters))
        GMM = GaussianMixture(n_components = clusters, max_iter = 20000, n_init = 50).fit(truncatedFeatures)
        labels = GMM.predict(truncatedFeatures)
        bic = GMM.bic(truncatedFeatures)
        # # fit DBSCAN
        # labels = DBSCAN(eps = 1, min_samples = 5).fit_predict(truncatedFeatures)
        # plot
        plt.figure(figsize = (6,5))
        plt.scatter(embedded[:,0], embedded[:,1], marker = 'o',
                    c=labels, s=20, edgecolor='black', linewidths=0.5)
        plt.savefig('GMM_anim'+str(animal)+'_comp'+str(clusters)+'_bic'+str(bic)+'.png', dpi = 500)
        HCV_anim = compute_and_plot_HCV(labels, combined_trialtypes, combined_rewardinfo)
        HCV_all.append(HCV_anim)
        plt.close('all')

## plot HCV for all animals
plt.ioff()
n_groups = 4

h = HCV_all[:,0]
c = HCV_all[:,1]
v = HCV_all[:,2]

plt.figure(figsize=(4,4))
index = np.arange(n_groups)
bar_width = 0.26

rects1 = plt.bar(index, h, bar_width,
                 color='blue',
                 label='Homogeneity')
rects2 = plt.bar(index+bar_width, c, bar_width,
                 color='red',
                 label='Completeness')
# rects3 = plt.bar(index+2*bar_width, v, bar_width,
#                  color='r',
#                  label='V-Measure')
plt.xlabel('Animal')
plt.ylabel('Scores')
plt.ylim(0,1)
plt.xticks(index + bar_width/2, ('1', '2', '3', '4'))
plt.legend(frameon = False)
plt.savefig('HCV_scores_all_animals.png', dpi = 500)






################### SUPERVISED PREDICTION  ##################

# SVM (correct vs incorrect?)

# combine features
mode = '4classes'
# repeats = 10
repeats = 1
animals = [1,2,3,4]
features_frames = list(range(1, 61))

if mode == '4classes': classes = 4
else: classes = 2

train_precision = np.zeros((repeats, len(animals), len(features_frames), classes))
train_recall = np.zeros((repeats, len(animals), len(features_frames), classes))
test_precision = np.zeros((repeats, len(animals), len(features_frames), classes))
test_recall = np.zeros((repeats, len(animals), len(features_frames), classes))
train_acc = np.zeros((repeats, len(animals), len(features_frames)))
test_acc = np.zeros((repeats, len(animals), len(features_frames)))


for anim_count, animal in enumerate(animals):
    print()
    print('*** ANIMAL: '+str(animal))
    for f_count, nb_frames in enumerate(features_frames): # truncate features
        print('* nb_frames = '+str(nb_frames))
        combined_features, corresponding_sess, \
        combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
        centroids = np.array(combined_features)[:, 0:2*nb_frames]
        orientations = np.array(combined_features)[:, 200:200+nb_frames]
        # combine features
        trial_type = combined_trialtypes
        X_all = np.concatenate((centroids,orientations), axis = 1)
        Y_all = np.array(combined_rewardinfo)[:,1]
        # remove trials where reward = 0 (timed out)
        zero_idxs = np.where(Y_all == 0)
        X = np.delete(X_all, zero_idxs, 0)
        Y = np.delete(Y_all, zero_idxs, 0)
        trials = np.delete(trial_type, zero_idxs, 0)
        if mode == '4classes':
            rewardRecord = Y
            Y_temp = []
            for stim, re in zip(trials, rewardRecord):
                if stim == 0 and re == 1:
                    Y_temp.append(1)
                elif stim == 0 and re == -1:
                    Y_temp.append(2)
                elif stim == 1 and re == 1:
                    Y_temp.append(3)
                elif stim == 1 and re == -1:
                    Y_temp.append(4)
        Y = np.array(Y_temp)
        # split data
        for trial in range(repeats):
            print()
            print('* Trial: '+str(trial+1))
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
            # train
            for c in [0.7]:
                print('* C = '+str(c))
                clf = svm.SVC(C = c, kernel = 'linear', verbose=2)
                clf.fit(X_train, Y_train)
                # accuracies
                train_score = clf.score(X_train, Y_train)
                test_score = clf.score(X_test, Y_test)
                # save scores
                train_acc[trial, anim_count, f_count] = train_score
                test_acc[trial, anim_count, f_count] = test_score
                # accuracies
                Y_train_predicted = clf.predict(X_train)
                Y_test_predicted = clf.predict(X_test)
                # compute precision / recall
                train_precision[trial, anim_count, f_count] = precision_score(Y_train, Y_train_predicted, average=None)
                train_recall[trial, anim_count, f_count] = recall_score(Y_train, Y_train_predicted, average=None)
                test_precision[trial, anim_count, f_count] = precision_score(Y_test, Y_test_predicted, average=None)
                test_recall[trial, anim_count, f_count] = recall_score(Y_test, Y_test_predicted, average=None)
                # report precision / recall
                print()
                print('Train')
                print(classification_report(Y_train, Y_train_predicted))
                print('Test')
                print(classification_report(Y_test, Y_test_predicted))
                print()
                # save as np array
                np.savez('SVM_run_multi_trial_4class',
                         train_acc = train_acc,
                         test_acc = test_acc,
                         train_precision = train_precision,
                         test_precision = test_precision,
                         train_recall = train_recall,
                         test_recall = test_recall)

##################
SVM_data = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/'
            'Mouse_tracking_analysis/SVM_run_multi_trial_4class.npz')
train_acc = SVM_data['train_acc']
test_acc = SVM_data['test_acc']
train_precision = SVM_data['train_precision']
test_precision = SVM_data['test_precision']
train_recall = SVM_data['train_recall']
test_recall = SVM_data['test_recall']
# plot precision / recall / f1 (ON TEST DATA!)
plt.ioff()
for anim in [1,2,3,4]:
    anim_precision = test_precision[:,anim-1,:,:]
    anim_recall = test_recall[:,anim-1,:,:]
    f1_score = 2 * anim_precision * anim_recall / (anim_precision + anim_recall)
    f1_score = np.nan_to_num(f1_score)
    for mode in ['precision', 'recall', 'f1']:
        plt.figure(figsize=(3, 2.7))
        if mode == 'precision':
            avg_acc = np.average(anim_precision, axis=0)
            acc_error = stats.sem(anim_precision, axis=0)
            ylabel = 'Precision'
        elif mode == 'recall':
            avg_acc = np.average(anim_recall, axis=0)
            acc_error = stats.sem(anim_recall, axis=0)
            ylabel = 'Recall'
        elif mode == 'f1':
            avg_acc = np.average(f1_score, axis = 0)
            acc_error = stats.sem(f1_score, axis = 0)
            ylabel = 'F1 Score'
        else:
            break
        top_err, bottom_err = avg_acc + acc_error, avg_acc - acc_error
        categories = ['L, L', 'L, R', 'R, R', 'R, L']
        colors = ['C0', 'darkgreen', 'darkorange', 'C3']
        # transpose axis
        avg_acc, top_err, bottom_err = np.transpose(avg_acc), np.transpose(top_err), np.transpose(bottom_err)
        for cat, avg_trace, top_err_trace, low_error_trace, color in \
                zip(categories, avg_acc, top_err, bottom_err, colors):
            plt.fill_between(np.arange(1,61), low_error_trace, top_err_trace,
                             interpolate=True, alpha=0.2, facecolor = color)
            plt.plot(np.arange(1,61), avg_trace, label=cat, color = color)
        plt.xlim(1,60)
        plt.ylim(0,1)
        plt.xlabel('Time (Frames)')
        plt.ylabel(ylabel)
        plt.legend(frameon = False, fontsize = 8)
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig('SVM_4class_'+mode+'_anim'+str(anim)+'.png', dpi = 500,
                    bbox_inches = 'tight')
        plt.close('all')

# plot accuracy for each animal on separate plots
plt.ioff()
plt.figure(figsize=(3, 2.7))
for anim in [1,2,3,4]:
    anim_data = test_acc[:,anim-1,:]
    avg_acc = np.average(anim_data, axis = 0)
    acc_error = stats.sem(anim_data, axis = 0)
    top_err, bottom_err = avg_acc + acc_error, avg_acc - acc_error
    plt.fill_between(np.arange(1,61), bottom_err, top_err,
                     interpolate=True, alpha=0.2)
    plt.plot(np.arange(1,61), avg_acc, label='Animal ' + str(anim))
plt.xlim(1, 60)
plt.ylim(0.5, 1)
plt.xlabel('Time (Frames)')
plt.ylabel('Test Accuracy')
plt.legend(frameon=False, fontsize=8)
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('test_acc_SVM_multi_trial_4class.png', dpi = 500,
                    bbox_inches = 'tight')
plt.close('all')

# Compile features for all animals for SVM classifier
X_all = []
Y_all = []
trials_all = []
rewards_all = []
nb_frames = 45 # previously tried 60
for animal in [1,2,3,4]:
    combined_features, corresponding_sess, \
    combined_startframes, combined_trialtypes, combined_rewardinfo = combine_tSNE_features(animal, all_sessions)
    centroids = np.array(combined_features)[:, 0:2 * nb_frames]
    orientations = np.array(combined_features)[:, 200:200 + nb_frames]
    # combine features
    X = np.concatenate((centroids, orientations), axis=1)
    trial_type = combined_trialtypes
    rewards = np.array(combined_rewardinfo)[:,1]
    # remove trials where reward = 0 (timed out)
    Y = np.ones(np.shape(X)[0]) * animal
    # append
    X_all.append(X)
    Y_all.append(Y)
    trials_all.append(trial_type)
    rewards_all.append(rewards)

X_all = np.concatenate(X_all, axis = 0)
Y_all = np.concatenate(Y_all, axis = 0)
trials_all = np.concatenate(trials_all, axis = 0)
rewards_all = np.concatenate(rewards_all, axis = 0)
# reshape and add trial info and reward info into X feature
# trials_all = np.reshape(trials_all, (np.size(trials_all),1))
# rewards_all = np.reshape(rewards_all, (np.size(trials_all),1))
# X_all = np.concatenate((X_all,trials_all,rewards_all), axis = 1)

#### TRAIN SVM

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2)
clf = svm.SVC(C=1, kernel='linear', verbose = True)
clf.fit(X_train, Y_train)
# accuracies
print('Train score: '+str(clf.score(X_train, Y_train)))
print('Test score: '+str(clf.score(X_test, Y_test)))
# report precision / recall
Y_train_predicted = clf.predict(X_train)
Y_test_predicted = clf.predict(X_test)
print('Training Data')
print(classification_report(Y_train, Y_train_predicted))
print('Testing Data')
print(classification_report(Y_test, Y_test_predicted))
print()

# save model
joblib.dump(clf, 'SVM_classify_animal_linear_includingtrialrewardinfo.pk1')

### PCA on ALL animals
PCArun = PCA(n_components = 4, whiten = False)
embedded = PCArun.fit_transform(X_all)
# ICA on all animals
ICArun = FastICA(n_components = 4)
ICArun.fit(X_all)

mode = 'PCA'

# PCA component analysis (separate animals)
for animal in [1,2,3,4]:
    nb_frames = 31
    combined_features, _, _, _, _ = combine_tSNE_features(animal, all_sessions)
    centroids = np.array(combined_features)[:, 0:2 * nb_frames]
    orientations = np.array(combined_features)[:, 200:200 + nb_frames]
    # combine features
    X = np.concatenate((centroids, orientations), axis=1)
    # fit PCA
    if mode == 'PCA':
        PCArun = PCA(n_components = 4, whiten = False)
        PCArun.fit(X)
        components = PCArun.components_
        explained_var = PCArun.explained_variance_ratio_
    # fit ICA
    if mode == 'ICA':
        ICArun = FastICA(n_components = 4, max_iter = 10000000)
        ICArun.fit(X)
        components = ICArun.components_
        S = ICArun.fit_transform(X)
        kurtosis = np.array(pd.DataFrame(S).kurt(axis=0))
    # analysis
    ########## plot PCA components
    positions = np.reshape(components[:,0:2*nb_frames], (4,nb_frames,2))
    orientations = components[:,2*nb_frames:]
    plt.figure(figsize=(3.2,2.9))
    for count, (traj, orient) in enumerate(zip(positions, orientations)):
        if mode == 'PCA':
            label = 'Component '+str(count+1) +', expvar = ' + str(explained_var[count])
        else:
            label = 'Component ' + str(count + 1)+', kurtosis = ' + str(kurtosis[count])
        if count < 3:
            # only plot first 3 components
            plt.plot(traj[:, 0], traj[:, 1], label= label, marker='x')
    plt.legend(frameon = False)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-0.2,0.35)
    if mode == 'ICA':
        plt.savefig('ICA_component_analysis_trajectory_anim'+str(animal)+'.png')
    else:
        plt.savefig('PCA_component_analysis_trajectory_anim'+str(animal)+'.png')
    ## PLOT ORIENTATIONS
    plt.figure(figsize= (3.2,2.9))
    for count, orient in enumerate(orientations):
        if mode == 'PCA':
            label = str(count+1) +': ' + str(round(explained_var[count], 2))
        else:
            label = str(count+1)+', kurtosis = ' + str(round(kurtosis[count], 2))
        if count < 3:
            # only plot first 3 components
            plt.plot(orient, label = label)
    plt.ylim(-0.2,0.35)
    plt.xlim(0,nb_frames-1)
    plt.xlabel('Time (Frames)')
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(np.arange(0, nb_frames+1, step=10), size=8)
    plt.yticks(size = 8)
    loc = ['upper right',
           'upper right',
           'upper right',
           'upper right']
    plt.legend(frameon = False, fontsize = 7, loc = loc[animal-1])
    if mode == 'ICA':
        plt.savefig('ICA_component_analysis_orientation_anim'+str(animal)+'.png', dpi = 500)
    else:
        plt.savefig('PCA_component_analysis_orientation_anim'+str(animal)+'.png', dpi = 500)
    # PLOT DISTANCE
    dists = compute_euc_dist(positions, centerport = None)
    plt.figure(figsize= (3.2,2.9))
    for count, dist in enumerate(dists):
        if mode == 'PCA':
            label = str(count+1) +': ' + str(round(explained_var[count], 2))
        else:
            label = str(count+1)+' kurtosis = ' + str(kurtosis[count])
        if count < 3:
            # only plot first 3 components
            plt.plot(dist, label = label)
    plt.legend(frameon = False, fontsize = 7)
    plt.xlabel('Time (Frames)')
    plt.xticks(np.arange(0, nb_frames+1, step=10), size=8)
    plt.yticks(size = 8)
    plt.xlim(0,nb_frames-1)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if mode == 'ICA':
        plt.savefig('ICA_component_analysis_distance_anim'+str(animal)+'.png',
                    dpi = 500, bbox_inches = 'tight')
    else:
        plt.savefig('PCA_component_analysis_distance_anim'+str(animal)+'.png',
                    dpi = 500, bbox_inches = 'tight')



############ plot PCA embedding (ALL animals one PLOT)
plt.ioff()
plt.figure(figsize=(13,12))
colormap = []
for anim in Y_all:
    if anim == 1:
        colormap.append('green')
    elif anim == 2:
        colormap.append('red')
    elif anim == 3:
        colormap.append('yellow')
    elif anim == 4:
        colormap.append('blue')

plt.scatter(embedded[:,0],embedded[:,1], c = colormap, s = 23,
            edgecolor = 'black',linewidth = 0.6)
anim1 = patches.Patch(color='green', label='Animal 1')
anim2 = patches.Patch(color='red', label='Animal 2')
anim3 = patches.Patch(color='yellow', label='Animal 3')
anim4 = patches.Patch(color='blue', label='Animal 4')

plt.legend(frameon=False, handles=[anim1, anim2, anim3, anim4], loc='upper left')
plt.savefig('All_Animals_tSNE.pdf', dpi = 500)



############ k means
# from sklearn.cluster import KMeans
#
# def compute_clusters(embedding):
#     cluster_kmean = KMeans(n_clusters=4).fit(embedding)
#     return cluster_kmean.labels_
#
# embeddings = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/tSNE_learnlight_animal4_window90/embedded_animal4.npy')
# labels = compute_clusters(embeddings)
#
# plt.figure(figsize=(8, 7))
# plt.scatter(embeddings[:,0], embeddings[:,1], cmap='rainbow', marker='o',
#             c=labels, s=15, edgecolor='black', linewidths=0.9)
# plt.savefig('test.pdf')


####### analyze trajectory data ################

def trajmap_animation_tSNE(embedding_dir):
    '''
    :param animal: 1,2,3 or 4
    :param embedding: tSNE embedding directory
    :param features: features to be fed into tSNE
    :return:
    '''
    import matplotlib.gridspec as gridspec
    plt.ioff()
    animal = int(embedding_dir[-5])
    embedding = np.load(embedding_dir)
    combined_features, _, _, _, _ = combine_tSNE_features(animal, all_sessions)
    ######### parameters for tiling space #############
    pref_blocks_across = 7 # number of blocks NOT OVERLAPPIN = 8 * 8 = 64
    step_prop = 3/4 # proportion of box to tile step right / down
    #####################################
    block_width = int((max(embedding[:,0]) - min(embedding[:,0])) / pref_blocks_across)
    block_height = int((max(embedding[:,1]) - min(embedding[:,1])) / pref_blocks_across)
    block_W_overlap = step_prop * block_width
    black_H_overlap = step_prop * block_height
    ## create all box coordinates. each box is defined by 2 x's y's, representing xmin, xmax, ymin, ymax
    xmins = np.arange(min(embedding[:,0]), max(embedding[:,0]) - block_width + 1, block_W_overlap)
    xmins = xmins.reshape(np.shape(xmins)[0],1)
    xmaxs = xmins + block_width
    xmaxs[-1] = max(embedding[:,0]) # set to max x value
    xs = np.concatenate((xmins,xmaxs),axis = 1)
    ymins = np.arange(min(embedding[:,1]), max(embedding[:,1]) - block_height + 1, black_H_overlap)
    ymins = ymins.reshape(np.shape(ymins)[0],1)
    ymaxs = ymins + block_height
    ymaxs[-1] = max(embedding[:,1]) # set to max y value
    ys = np.concatenate((ymins,ymaxs),axis = 1)
    ## loop through all the x and y pairs
    frame_len = 30
    combined_features = np.array(combined_features)
    all_averaged_traj = np.zeros((np.shape(xs)[0], np.shape(ys)[0], np.shape(combined_features)[1]))
    for x_count, x_pair in enumerate(xs):
        for y_count, y_pair in enumerate(ys):
            embedding_x = embedding[:,0]
            embedding_y = embedding[:,1]
            idxs = np.where((embedding_x >= x_pair[0]) & (embedding_x <= x_pair[1]) &
                            (embedding_y >= y_pair[0]) & (embedding_y <= y_pair[1]))[0]
            if idxs.size > 5: # minimum number of points in block to proceed with averaging
                features = combined_features[idxs]
                features_avg = np.average(features, axis = 0)
                all_averaged_traj[x_count,y_count,:] = features_avg

    ## plot animation of centroid and orientation
    print('Animating trajectories on tSNE embedding space for animal '+str(animal))
    nb_rows, nb_columns = np.shape(all_averaged_traj)[1], np.shape(all_averaged_traj)[0]
    fig = plt.figure(figsize=(nb_columns, nb_rows))
    grid = gridspec.GridSpec(nb_rows, nb_columns)
    grid.update(wspace=0, hspace=0)
    video = []
    print('Rendering animation..')
    # centroid_list = [] # not for any program function
    arrow_length = 0.5 # adjust
    for frame_nb in range(frame_len):
        frame_i = []
        block = 0
        skip = 1
        for row in range(nb_rows-1,-1,-skip): # negative iteration because plotting from top left corner
            for col in range(0,nb_columns,skip):
                ax = plt.subplot(grid[block])
                ax.axis('off')
                ax.set_xlim(250, 575)
                ax.set_ylim(160, 450)
                centroid_x = all_averaged_traj[col, row, frame_nb*2]
                centroid_y = all_averaged_traj[col, row, frame_nb*2+1]
                angle = all_averaged_traj[col, row, 200+frame_nb]
                angle_radian = angle * np.pi / 180
                dx, dy = np.cos(angle_radian) * arrow_length, np.sin(angle_radian) * arrow_length
                if centroid_x == 0 and centroid_y == 0:
                    obj, = ax.plot(centroid_x, centroid_y, marker='o', c='white')
                else:
                    obj = ax.arrow(centroid_x, centroid_y, dx, dy, color = 'red', width = 17)
                # frame_i.extend((centroid))
                frame_i.append(obj)
                block += 1
        video.append(frame_i)
    # compile animation
    Anim = animation.ArtistAnimation(fig, video, interval=2000/30, blit=True) # currently set to 1/2 speed
    print('Saving video..')
    Anim.save('learnlight_tSNE_trajectory_animation_animal'+str(animal)+'.mp4')

# # RUN tSNE animation
# for animal in [1,2,3,4]:
#     embedding_dir = '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
#                     'tSNE_learnlight_animal'+str(animal)+'_window135/embedded_animal'+str(animal)+'.npy'
#     trajmap_animation_tSNE(embedding_dir)
# RUN LDA animation
for animal in [1,2,3,4]:
    embedding_dir = '/home/tzhang/Dropbox/Caltech/' \
                    'VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                    'Manuscript_Figures_(new)/LDA/LDA_embedding30_anim_rt'+\
                    str(animal)+'.npy'
    trajmap_animation_tSNE(embedding_dir)


def trajmap_animation_PCA(embedding_dir):
    '''
    :param animal: 1,2,3 or 4
    :param embedding: tSNE embedding directory
    :param features: features to be fed into tSNE
    :return:
    '''
    plt.ioff()
    animal = int(embedding_dir[-5])
    embedding = np.load(embedding_dir)
    combined_features, _, _, _, _ = combine_tSNE_features(animal, all_sessions)
    ######### parameters for tiling space #############
    pref_blocks_across = 11 # number of blocks NOT OVERLAPPIN = 8 * 8 = 64
    step_prop = 3/4 # proportion of box to tile step right / down
    #####################################
    block_width = int((max(embedding[:,0]) - min(embedding[:,0])) / pref_blocks_across)
    block_height = int((max(embedding[:,1]) - min(embedding[:,1])) / pref_blocks_across)
    block_W_overlap = step_prop * block_width
    black_H_overlap = step_prop * block_height
    ## create all box coordinates. each box is defined by 2 x's y's, representing xmin, xmax, ymin, ymax
    xmins = np.arange(min(embedding[:,0]), max(embedding[:,0]) - block_width + 1, block_W_overlap)
    xmins = xmins.reshape(np.shape(xmins)[0],1)
    xmaxs = xmins + block_width
    xmaxs[-1] = max(embedding[:,0]) # set to max x value
    xs = np.concatenate((xmins,xmaxs),axis = 1)
    ymins = np.arange(min(embedding[:,1]), max(embedding[:,1]) - block_height + 1, black_H_overlap)
    ymins = ymins.reshape(np.shape(ymins)[0],1)
    ymaxs = ymins + block_height
    ymaxs[-1] = max(embedding[:,1]) # set to max y value
    ys = np.concatenate((ymins,ymaxs),axis = 1)
    ## loop through all the x and y pairs
    frame_len = 45
    combined_features = np.array(combined_features)
    all_averaged_traj = np.zeros((np.shape(xs)[0], np.shape(ys)[0], np.shape(combined_features)[1]))
    for x_count, x_pair in enumerate(xs):
        for y_count, y_pair in enumerate(ys):
            embedding_x = embedding[:,0]
            embedding_y = embedding[:,1]
            idxs = np.where((embedding_x >= x_pair[0]) & (embedding_x <= x_pair[1]) &
                            (embedding_y >= y_pair[0]) & (embedding_y <= y_pair[1]))[0]
            if idxs.size > 5: # minimum number of points in block to proceed with averaging
                features = combined_features[idxs]
                features_avg = np.average(features, axis = 0)
                all_averaged_traj[x_count,y_count,:] = features_avg

    ## plot animation of centroid and orientation
    print('Animating trajectories on tSNE embedding space for animal '+str(animal))
    nb_rows, nb_columns = np.shape(all_averaged_traj)[1], np.shape(all_averaged_traj)[0]
    fig = plt.figure(figsize=(nb_columns, nb_rows))
    grid = gridspec.GridSpec(nb_rows, nb_columns)
    grid.update(wspace=0, hspace=0)
    video = []
    print('Rendering animation..')
    # centroid_list = [] # not for any program function
    arrow_length = 0.5 # adjust
    for frame_nb in range(frame_len):
        frame_i = []
        block = 0
        skip = 1
        for row in range(nb_rows-1,-1,-skip): # negative iteration because plotting from top left corner
            for col in range(0,nb_columns,skip):
                ax = plt.subplot(grid[block])
                ax.axis('off')
                ax.set_xlim(250, 575)
                ax.set_ylim(160, 450)
                centroid_x = all_averaged_traj[col, row, frame_nb*2]
                centroid_y = all_averaged_traj[col, row, frame_nb*2+1]
                angle = all_averaged_traj[col, row, 200+frame_nb]
                angle_radian = angle * np.pi / 180
                dx, dy = np.cos(angle_radian) * arrow_length, np.sin(angle_radian) * arrow_length
                if centroid_x == 0 and centroid_y == 0:
                    obj, = ax.plot(centroid_x, centroid_y, marker='o', c='white')
                else:
                    obj = ax.arrow(centroid_x, centroid_y, dx, dy, color = 'red', width = 17)
                # frame_i.extend((centroid))
                frame_i.append(obj)
                block += 1
        video.append(frame_i)
    # compile animation
    Anim = animation.ArtistAnimation(fig, video, interval=1000/30, blit=True) # currently set to 1/2 speed
    print('Saving video..')
    Anim.save('learnlight_PCA_trajectory_animation_animal'+str(animal)+'.mp4')
# RUN animation
for animal in [1,2,3,4]:
    embedding_dir = '/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                    'PCA_learnlight_animal'+str(animal)+'_window135/embedded_PCA_animal'+str(animal)+'.npy'
    trajmap_animation_PCA(embedding_dir)

######### generate distance plots #########

animal = 1
embedding = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                    'tSNE_learnlight_animal1_window135/embedded_animal1.npy')
clusters_coord = {
    # 'CLUSTER': [x1, x2, y1, y2]
    'LL': np.array([[45, 80, -20, 40], [10, 45, 25, 40]]),
    'RR': np.array([[-70, -40, -40, -5], [-40, 0, 0, 40]]),
    'LR': np.array([[-35, 10, -50, 0]]),
    'RL': np.array([[10, 45, -10, 18]])
}
plot_dist_to_nosepoke(animal, clusters_coord, embedding)


####################
animal = 2
embedding = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                    'tSNE_learnlight_animal2_window135/embedded_animal2.npy')
clusters_coord = {
    # 'CLUSTER': [x1, x2, y1, y2]
    'LL': np.array([[-40, -20, -10, 40]]),
    'RR': np.array([[21, 40, -30, 20]]),
    'LR': np.array([[5, 17, -5, 12]]),
    'RL': np.array([[-20, 4, -20, 10]])
}

plot_dist_to_nosepoke(animal, clusters_coord, embedding)


####################
animal = 3
embedding = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                    'tSNE_learnlight_animal3_window135/embedded_animal3.npy')
clusters_coord = {
    # 'CLUSTER': [x1, x2, y1, y2]
    'LL': np.array([[-40, 20, 40, 80], [20, 50, 20, 60]]),
    'RR': np.array([[-60, -30, -60, -20], [-30, 24, -80, -37]]),
    'LR': np.array([[20, 60, -40, 0]]),
    'RL': np.array([[-50, 0, -5, 40]]),
    'Mixed (5th cluster)': np.array([[-17, 11, -37, 5]])
}
plot_dist_to_nosepoke(animal, clusters_coord, embedding)

###################
animal = 4
embedding = np.load('/home/tzhang/Dropbox/Caltech/VisionLab/Mouse_Project/Mouse_tracking_analysis/' \
                    'tSNE_learnlight_animal4_window135/embedded_animal4.npy')
clusters_coord = {
    # 'CLUSTER': [x1, x2, y1, y2]
    'LL': np.array([[0, 70, 5, 50], [40, 79, -40, 5]]),
    'RR': np.array([[-65, -23, -40, 40]]),
    'LR': np.array([[-30, -5, 0, 28]]),
    'RL': np.array([[0, 30, -70, -30]]),
    'Mixed (middle cluster)': np.array([[-8, 30, -20, 5]])
}
plot_dist_to_nosepoke(animal, clusters_coord, embedding)


#################### reference code
def animate_avg_trajectories(trajectories):
    '''
    :param trajectories: has shape (block nb x-direction, black nb y-direction, features)
    :return:
    '''

    def animate_multi_view(self, start, duration, centroids, keypoints, skip_frames = 1,
                           interval = 1000/30, repeat = False, save = False, animate = True, video = True):
        shift = self.shift
        box = math.ceil(math.sqrt(len(start)))
        fig = plt.figure(figsize = (10,10))
        gs1 = gridspec.GridSpec(box,box)
        gs1.update(wspace=0, hspace=0)
        ims = []
        filename = self.Video.name + '_' + str(start) + '_' + str(duration) + '.mp4'
        print()
        print('Rendering video: ' + filename)
        for i in range(0, duration, skip_frames):
            ims_i = []
            for j in range(len(start)):
                # CHOOSE SUBPLOT
                ax = plt.subplot(gs1[j])
                ax.axis('off')
                xmin, xmax = 0,
                ymin, ymax = []
                ax.set_xlim(0, 800)
                ax.set_ylim(0, 650)
                # PLOT CENTROID / NOSE
                x1 = centroids[start[j] + i - shift, 0]
                y1 = centroids[start[j] + i - shift, 1]
                x2 = keypoints[start[j] + i - shift, 0, 0]
                y2 = keypoints[start[j] + i - shift, 1, 0]
                centroid, = ax.plot(x1, y1, marker='o', c = 'blue')
                nose, = ax.plot(x2, y2, marker='o', c = 'green')
                # LABELS
                label_start_frame = ax.annotate(start[j], (4, 45), fontsize=12, color = 'white')
                label_curr_frame = ax.annotate(start[j]+i, (4, 85), fontsize=9, color='white')
                # COMBINE ALL ACTIONS
                if video:
                    # PLOT FRAME
                    img_i_j = self.Video.capture_frame(start[j]+i)
                    # pdb.set_trace()
                    frame_i_j = ax.imshow(img_i_j)

                    ims_i.extend((frame_i_j, centroid, nose, label_start_frame, label_curr_frame))

                else: ims_i.extend((centroid, nose, label_start_frame, label_curr_frame))
            ims.append(ims_i)
        # Ac
        Anim = animation.ArtistAnimation(fig, ims, interval = interval, blit = True, repeat = repeat, repeat_delay=3000)
        if save:
            print('Saving video: ' + filename)
            Anim.save(filename)
        if animate: return Anim










