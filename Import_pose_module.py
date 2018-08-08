import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import math
from matplotlib import animation
import pdb
import scipy.io as sio
import plotly
plotly.tools.set_credentials_file(username='tonyzhang', api_key='6zoZzkCL9hqQSWDOj0xq')
import plotly.plotly as py
from plotly.graph_objs import *
import matplotlib.patches as patches
from numpy import diff, where, split
import glob
import seaborn as sns
import math
sns.set() # warning: this sets the ticks to invisible
sns.set_style("white")
import os

class Load_Pose():

    def __init__(self, pose_dir):
        # with open(root_dir + '/output/' + vid_name + '_pose.json', 'r') as f:
        #     POSE = json.load(f)
        with open(pose_dir, 'r') as f:
            POSE = json.load(f)
        self.bscores = np.array(POSE['bscores'])
        self.bbox = np.array(POSE['bbox'])
        self.kp_confidence = np.array(POSE['scores'])  # keypoint confidence scores
        self.keypoints = np.array(POSE['keypoints'])  # locations of 7 points characterizing mouse

class Pair_matfile_with_video():
    '''
    Video is stored at the root_dir level
    matfile is stored at the root_dir/Matfiles level
    '''
    def __init__(self, root_dir):
        self.matfile_dir = root_dir+'/Matfiles'
        self.video_list = glob.glob(root_dir + '/*.mj2')
        self.matfile_list = glob.glob(self.matfile_dir + '/*.mat')
        self.pairs = []
        if len(self.video_list) != len(self.matfile_list):
            print('ERROR: video / matfile list length mismatch')

        self.pair()

    def pair(self):
        # PAIRING HAPPENS BELOW
        for vid in self.video_list:
            ending = '/*' + vid[-19:-11] + '?' + vid[-10:-6] + '??.mat'
            paired_matfile = glob.glob(self.matfile_dir + ending)
            if len(paired_matfile) > 0:
                self.pairs.append([vid, paired_matfile[0]])
            else: # conduct secondary distance based pairing
                min = 10 # arbitrarily large
                for matfile in (self.matfile_list):
                    # first: assert 1) same animal 2) same day
                    if matfile[-42:-11] == vid[-42:-11]:
                        matfile_hour = int(matfile[-8:-6])
                        video_hour = int(vid[-8:-6])
                        diff = abs(video_hour - matfile_hour)
                        if diff < min:
                            min = diff
                            paired_matfile = matfile
                self.pairs.append([vid, paired_matfile])
        self.pairs = np.array(self.pairs) # convert to numpy array for ease of indexing
        return self.pairs



class VideoCapture():

    def __init__(self, root_dir, vid_name):
        self.cap = cv2.VideoCapture(root_dir + vid_name)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames
        self.TRIALS = []
        self.LED_flash = []
        self.nosepoke_light = []
        self.name = vid_name

    def capture_frame(self,frame_N):
        self.cap.set(1,frame_N)
        ret, frame = self.cap.read()
        return frame

    def compute_first_trial_frame(self): # frame = single RGB frame from the video
        threshold = 130  # calibrated based on video: Animal4_LearnWNandLight_20171028T202815
        print('Frames to process: ' + str(self.frames))
        for i in range(self.frames):
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            L_brightness = self.compute_luminance(frame[145:250, 535:610])  # left cue block
            # print(L_brightness)
            R_brightness = self.compute_luminance(frame[353:458, 535:610])  # right cue block
            # print(R_brightness)
            if L_brightness > threshold:
                self.TRIALS.append([i, 0]) # i = frame, 0 = left
                break
            elif R_brightness > threshold:
                self.TRIALS.append([i, 1]) # i = frame, 1 = right
                break

    def compute_visual_CUE_times(self): # frame = single RGB frame from the video
        threshold = 150 ## test value
        trial_max = 200
        print('Frames to process: ' + str(self.frames))
        repeat = False
        for i in range(self.frames):
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            L_brightness = self.compute_luminance(frame[145:250, 535:610])  # left cue block
            R_brightness = self.compute_luminance(frame[353:458, 535:610])  # right cue block
            if L_brightness > threshold:
                if repeat == False:
                    self.TRIALS.append([i, 0]) # i = frame, 0 = left
                    print('Frame: ' + str(i) + '\nTrial: Left')
                    if len(self.TRIALS) == trial_max: break
                    repeat = True
            elif R_brightness > threshold:
                if repeat == False:
                    self.TRIALS.append([i, 1]) # i = frame, 1 = right
                    print('Frame: ' + str(i) + '\nTrial: Right')
                    if len(self.TRIALS) == trial_max: break
                    repeat = True
            else: repeat = False
        self.TRIALS = np.array(self.TRIALS)

    def compute_LED_times(self, vid_name_path, limited = True, nb_trials = 5, detectstim = False):
        # limited = only compute times for first 5 and last 5 flashes
        '''
        :param vid_name_path:
        :param limited:
        :param nb_trials:
        :param detectstim: whether or not to also detect visual port lights in addition to LED
        :return:
        '''
        LED_threshold = 150
        nosepoke_threshold = 170
        print('Frames to process: ' + str(self.frames))
        i = 0

        if limited:
            print('Setting: limited (5 flashes from start / end)')
            print('Computing 1st 5 flashes..')
            while len(self.LED_flash) < nb_trials:
                frame = self.capture_frame(i)
                # pdb.set_trace()
                LED_brightness = self.compute_luminance(frame[320:370, 665:710])
                if LED_brightness > LED_threshold:
                    self.LED_flash.append(i) # i = frame
                    print('Detected LED flash: ' + str(i))
                    i += 2
                i += 1
            i = self.frames - 1 # set to last frame of video
            print('Computing last 5 flashes..')
            while len(self.LED_flash) < 2*nb_trials:
                frame = self.capture_frame(i)
                LED_brightness = self.compute_luminance(frame[320:370, 665:710])
                # print(i) # for debugging
                if LED_brightness > LED_threshold:
                    self.LED_flash.append(i) # i = frame
                    print('Detected LED flash: ' + str(i))
                    i -= 2
                i -= 1

        else: # testing oisin's new method
            print('Setting: limited OFF (all flashes)')
            if detectstim:
                # also capture the nose poke port lights
                while i < self.frames: # i = frame
                    if i % 1000 == 0:
                        print('Progress: '+str(i))
                    ret, frame = self.cap.read()
                    if frame is not None:
                        LED_brightness = self.compute_luminance(frame[320:370, 665:710]) # position of LED. May need adjusting.
                        Lport_brightness = self.compute_luminance(frame[175:270, 525:600])
                        Rport_brightness = self.compute_luminance(frame[370:465, 525:600])
                        if LED_brightness > LED_threshold:
                            self.LED_flash.append(i)
                            print('Detected LED flash: ' + str(i))
                        if Lport_brightness > nosepoke_threshold:
                            self.nosepoke_light.append(i)
                            self.TRIALS.append('L') # 0 = LEFT
                            print('Detected left nosepoke light: ' + str(i))
                        elif Rport_brightness > nosepoke_threshold:
                            self.nosepoke_light.append(i)
                            self.TRIALS.append('R') # 1 = RIGHT
                            print('Detected right nosepoke light: ' + str(i))
                    i += 1 # increase frame number
            else:
                while i < self.frames:
                    if i % 100 == 0:
                        print('Progress: '+str(i))
                    ret, frame = self.cap.read()
                    if frame is not None:
                        LED_brightness = self.compute_luminance(frame[320:370, 665:710]) # position of LED. May need adjusting.
                        if LED_brightness > LED_threshold:
                            self.LED_flash.append(i) # i = frame
                            print('Detected LED flash: ' + str(i))
                    i += 1

        self.LED_flash = np.array(self.LED_flash)
        self.nosepoke_light = np.array(self.nosepoke_light)

        if limited: np.save(vid_name_path + '_LED_times_limited', {'LED_flash': self.LED_flash})
        else:
            if detectstim: np.save(vid_name_path + '_LED_and_nosepoke_flashes',
                                   {'LED_flash': self.LED_flash,
                                    'nosepoke_flashes': self.nosepoke_light,
                                    'trial_types': self.TRIALS})
            else: np.save(vid_name_path + '_LED_flashes_ALL', {'LED_flash': self.LED_flash})

    def compute_LED_luminence(self, vid_name_path):
        self.all_LED_lum = []
        print('Frames to process: ' + str(self.frames))
        for i in range(self.frames):
            if i % 100 == 0:
                print('Progress: '+str(i))
            frame = self.capture_frame(i)
            # LED_brightness = self.compute_luminance(frame[310:360, 705:750]) # position of LED. May need adjusting.
            LED_brightness = self.compute_luminance(frame[320:370, 665:710]) # new one tuned to yellow HDDs
            self.all_LED_lum.append(LED_brightness)
        self.all_LED_lum = np.array(self.all_LED_lum)
        np.save(vid_name_path + '_all_LED_lum', self.all_LED_lum)

    def compute_luminance(self, block):
        avg_brightness = np.average(block)
        return avg_brightness
        # block is a RGB patch in shape (x-dim, y-dim, 3)

    def show_clip(self, start, end, interval = 1000/30, repeat = False, save = False, label = False):
        ims = []
        fig = plt.figure("Animation")
        ax = fig.add_subplot(111)
        ax.axis('off')
        print('Animating..')
        for i in range(start, end+1):
            img = self.capture_frame(i)
            frame_i = ax.imshow(img)
            if label:
                label = ax.annotate(i, (10,40), color = 'white', size = 15)
                ims.append([frame_i, label])
            else:
                ims.append([frame_i])
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        Anim = animation.ArtistAnimation(fig, ims, interval = interval, blit = True, repeat = repeat)
        if save: Anim.save(str(start) + '_' + str(end) + '.mp4')
        return Anim


class Smooth_Keypoints():

    def __init__(self, Pose, window):
        self.window = window
        self.keypoints = Pose.keypoints
        self.kp_confidence = Pose.kp_confidence
        self.shift = self.window // 2

    def Weighted_Mov_Avg(self):
        window = self.window
        keypoints = self.keypoints
        kp_confidence = self.kp_confidence
        frames = np.shape(keypoints)[0]
        output = np.zeros((frames - window + 1, 2, 7))
        for i in range(frames - window):
            seq_i = keypoints[i:i + window]
            movavg_i_x = np.average(seq_i[:, :, 0, :], axis=0, weights=kp_confidence[i:i + window])
            movavg_i_y = np.average(seq_i[:, :, 1, :], axis=0, weights=kp_confidence[i:i + window])
            output[i, 0], output[i, 1] = movavg_i_x, movavg_i_y
        return output

    def Remove_Anomaly_Frames(self): # hard code in frame removal given change in xy coordinates greater than threshold
        keypoints = self.keypoints
        frames = np.shape(keypoints)[0]
        self.all_euc_dist = np.zeros(frames-1)
        dist_threshold = 600
        for i in range(1,frames):
            # compare current frame one previous frame
            kp_i = self.keypoints[i-1:i+1]
            # compute euclidean distance two frames
            euc_dist = norm(kp_i[1,0] - kp_i[0,0])
            self.all_euc_dist[i-1] = euc_dist

            if euc_dist > dist_threshold:
                # pass
                keypoints[i] = self.keypoints[i-1]
        return keypoints

        self.keypoints = keypoints

        # return all_euc_dist


class Analysis():

    # compute low dimensional info (centroid x-y coordinates & orientation in angles) for cluster analysis

    def __init__(self, keypoints, smoothing, posename): # pass smoothed keypoints in here
        self.keypoints = keypoints
        self.frames = np.shape(keypoints)[0]
        self.shift = smoothing.shift
        self.posename = posename

    def compute_centroid(self):
        self.centroids = np.average(self.keypoints, axis = 2)
        return self.centroids

    def compute_orientation(self):
        self.compute_centroid()
        nose_positions = self.keypoints[:,:,0]
        # connect centroid to keypoints. Output list of angles (taking centroid as origin), one for each frame.
        centroid_origin = nose_positions - self.centroids
        x = centroid_origin[:,0]
        y = centroid_origin[:,1]
        self.orientations = np.arctan2(y, x) * 180 / np.pi

    def tSNE(self, behavior_window, label): # behavior window = size of the window for analysis (in number of frames)
        self.compute_orientation()
        centroids = self.centroids
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        data =self.compile_data_tSNE_positions(end_frame, step, behavior_window)
        self.embedded = TSNE(n_components=2, verbose = True).fit_transform(data) # fit tSNE (sklearn)
        self.plot_tSNE(label, behavior_window, end_frame, step)

    def tSNE_only_orientation_and_velocity(self, behavior_window, label, perplexity):
        self.compute_speed_from_centroids()
        self.compute_orientation()
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        data = self.compile_data_tSNE_orientations(end_frame, step, behavior_window, orientations)
        self.embedded = TSNE(n_components=2, verbose = True, perplexity = perplexity).fit_transform(data) # fit tSNE
        self.plot_tSNE(label, behavior_window, end_frame, step)

    def tSNE_test_intertrial(self, behavior_window, frame_start, trial_TYPE, padding, speed = False, label = True, save = False):
        ''' positions only '''
        print()
        print('Computing tSNE for session: ' + self.posename[:-14])
        print('Properties:')
        print('Padding: ' + str(padding))
        self.compute_orientation()
        self.trial_TYPE = trial_TYPE
        centroids = self.centroids
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        if speed:
            self.compute_speed_from_centroids()
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_orientation_speed(frame_start,
                                                                                                  behavior_window,
                                                                                                  orientations)
        else:
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_positions(frame_start,
                                                                                          behavior_window,
                                                                                          orientations,
                                                                                          padding)
        self.tSNE_data = data
        self.embedded = TSNE(n_components=2, verbose = True).fit_transform(data) # fit tSNE (sklearn)
        self.plot_tSNE_intertrial(label, behavior_window, self.trial_history, self.trial_TYPE, step, save)

    def PCA_intertrial(self, behavior_window, label, trial_history, speed, padding): # TEST INTERTRIAL TSNE. group with other functions
        ''' positions only '''
        self.compute_orientation()
        centroids = self.centroids
        orientations = self.orientations
        end_frame = self.frames - behavior_window
        step = behavior_window//2
        if speed:
            self.compute_speed_from_centroids()
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_orientation_speed(trial_history, step, behavior_window, orientations)
        else:
            (data, behavior_window) = self.compile_intertrial_data_dimreduction_positions(trial_history, step, behavior_window, orientations, padding)
        self.PCA_embedded = PCA(n_components=2).fit_transform(data) # fit PCA (sklearn)
        pca = PCA(n_components=2).fit(data)
        self.PCA_cov_matrix = pca.get_covariance()

        self.plot_PCA_intertrial(label, behavior_window, trial_history, step)

    def compile_data_tSNE_positions(self, end_frame, step, behavior_window): # compile all data from all frames
        data = np.zeros((np.size(range(0, end_frame, step)), 3 * behavior_window))
        for count, frame in enumerate(range(0, end_frame, step)):
            centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
            orientations_flattened = self.orientations[frame:frame+behavior_window]
            data[count] = np.concatenate((centroids_flattened, orientations_flattened))
        return data

    def compile_data_tSNE_orientations(self, end_frame, step, behavior_window, orientations): # compile all data from all frames
        data = np.zeros((np.size(range(0,end_frame,step)), 2 * behavior_window))
        for count, frame in enumerate(range(0, end_frame, step)):
            speeds_flattened = self.speeds[frame:frame+behavior_window]
            orientations_flattened = orientations[frame:frame + behavior_window]
            data[count] = np.concatenate((speeds_flattened, orientations_flattened))
        return data

    # def compile_intertrial_data_dimreduction_positions(self, trial_history, step, behavior_window, orientations): # only compile relevant features after
    #     inter_trial_durations = trial_history[1:-1,0] - trial_history[0:-2,0]
    #     min_trial_duration = np.min(inter_trial_durations)
    #     behavior_window = min_trial_duration
    #     data = np.zeros((trial_history.shape[0], 3 * behavior_window))
    #     for i in range(data.shape[0]):
    #         frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
    #         centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
    #         orientations_flattened = orientations[frame:frame+behavior_window]
    #         data[i] = np.concatenate((centroids_flattened, orientations_flattened))
    #     return data, behavior_window

    def compile_intertrial_data_dimreduction_positions(self, trial_history, behavior_window, orientations, padding): # only compile relevant features after
        ## check when's the last trial captured by the video
        trial_history = trial_history[trial_history < self.frames] # get rid of frames that are beyond video length
        self.trial_history = trial_history
        self.trial_TYPE = self.trial_TYPE[:len(trial_history)]
        inter_trial_durations = trial_history[1:] - trial_history[:-1]
        last_trial_duration = self.frames - trial_history[-1]
        print('Last trial duration: ' + str(last_trial_duration))
        # check if the video ends too early:
        inter_trial_durations = np.append(inter_trial_durations, last_trial_duration)
        min_trial_duration = int(np.min(inter_trial_durations))
        print('Minimum trial duration: ' + str(min_trial_duration))
        if padding == False:
            behavior_window = min_trial_duration
        # pdb.set_trace()
        data = np.zeros((trial_history.shape[0], 3 * behavior_window))

        if padding and behavior_window > min_trial_duration:
            for i in range(data.shape[0]):
                frame = int(trial_history[i]) - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
                trial_duration = inter_trial_durations[i]
                if trial_duration < behavior_window:
                    centroids_flattened = np.ravel(self.centroids[frame:frame+trial_duration])
                    orientations_flattened = orientations[frame:frame+trial_duration]
                    data[i, :len(centroids_flattened)] = centroids_flattened
                    data[i, behavior_window*2:behavior_window*2+len(orientations_flattened)] = orientations_flattened
                else:
                    centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
                    orientations_flattened = orientations[frame:frame+behavior_window]
                    data[i] = np.concatenate((centroids_flattened, orientations_flattened))
                    #### FIX problem withh way concatenating features

        else: # no padding
            for i in range(data.shape[0]):

                frame = int(trial_history[i] - self.shift) # THIS IS DONE B/C OF SMOOTHING MISMATCH
                centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
                orientations_flattened = orientations[frame:frame+behavior_window]
                data[i] = np.concatenate((centroids_flattened, orientations_flattened))
        self.data_cluster_analysis = data
        return data, behavior_window

    def compile_intertrial_data_dimreduction_positions_old(self, trial_history, step, behavior_window, orientations, padding): # only compile relevant features after

        inter_trial_durations = trial_history[1:,0] - trial_history[0:-1,0]
        last_trial_duration = self.frames - trial_history[-1, 0]
        inter_trial_durations = np.append(inter_trial_durations, last_trial_duration)
        min_trial_duration = np.min(inter_trial_durations)
        if padding == False:
            behavior_window = min_trial_duration
        data = np.zeros((trial_history.shape[0], 3 * behavior_window))

        if padding:
            assert behavior_window > min_trial_duration
            for i in range(data.shape[0]):
                frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
                trial_duration = inter_trial_durations[i]
                if trial_duration < behavior_window:
                    centroids_flattened = np.ravel(self.centroids[frame:frame+trial_duration])
                    orientations_flattened = orientations[frame:frame+trial_duration]
                    data[i, :len(centroids_flattened)] = centroids_flattened
                    data[i, behavior_window*2:behavior_window*2+len(orientations_flattened)] = orientations_flattened
                else:
                    centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
                    orientations_flattened = orientations[frame:frame+behavior_window]
                    data[i] = np.concatenate((centroids_flattened, orientations_flattened))
        else: # no padding
            for i in range(data.shape[0]):
                frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
                centroids_flattened = np.ravel(self.centroids[frame:frame+behavior_window])
                orientations_flattened = orientations[frame:frame+behavior_window]
                data[i] = np.concatenate((centroids_flattened, orientations_flattened))
        self.data_cluster_analysis = data
        return data, behavior_window

    def compile_intertrial_data_dimreduction_orientation_speed(self, trial_history, step, behavior_window, orientations): # only compile relevant features after
        inter_trial_durations = trial_history[1:,0] - trial_history[0:-1,0]
        min_trial_duration = np.min(inter_trial_durations)
        behavior_window = min_trial_duration
        data = np.zeros((trial_history.shape[0], 2 * behavior_window))
        for i in range(data.shape[0]):
            frame = trial_history[i,0] - self.shift # THIS IS DONE B/C OF SMOOTHING MISMATCH
            speeds_flattened = self.speeds[frame:frame + behavior_window]
            orientations_flattened = orientations[frame:frame+behavior_window]
            data[i] = np.concatenate((speeds_flattened, orientations_flattened))
        return data

    def plot_tSNE(self, label, behavior_window, end_frame, step):
        embedded = self.embedded
        print('Plotting..')
        plt.figure()
        plt.title('t-SNE (window = '+str(behavior_window)+')')
        plt.scatter(embedded[:,0], embedded[:,1], color = 'red', s = 5)
        if label:
            for count, frame in enumerate(range(0, end_frame, step)):
                plt.annotate(frame,(embedded[count,0], embedded[count,1]), fontsize = 6)

    def plot_tSNE_intertrial(self, label, behavior_window, start_frames, trial_TYPE, step, save = False):
        embedded = self.embedded
        print('Plotting..')
        plt.figure(figsize=(10,9))
        plt.title('t-SNE (file: '+self.posename+', window = '+str(behavior_window)+')')
        # trial_types = trial_history[:,1]
        plt.scatter(embedded[:, 0], embedded[:, 1], cmap='bwr', c=trial_TYPE, s=18, edgecolor='black', linewidths=0.7)
        # plt.scatter(embedded[:, 0], embedded[:, 1], cmap='winter')
        if label:
            for count, frame in enumerate(start_frames):
                plt.annotate(frame,(embedded[count,0], embedded[count,1]), fontsize = 6)
                ''' note: this annotation is for the frames in the VIDEO, NOT indices of the centroids / orientations! 
                Remember to subtract by shift '''
        if save: plt.savefig(self.posename[:-14] + '.pdf')

    def plot_PCA_intertrial(self, label, behavior_window, trial_history, step):
        embedded = self.PCA_embedded
        print('Plotting..')
        plt.figure()
        plt.title('PCA Inter-trial (window = '+str(behavior_window)+')')
        trial_types = trial_history[:,1]
        frames = trial_history[:,0]
        # plt.scatter(embedded[:,0], embedded[:,1], cmap = 'winter', c = trial_types)
        plt.scatter(embedded[:, 0], embedded[:, 1], cmap='bwr', c=trial_types, s=18, edgecolor='black', linewidths=0.7)
        if label:
            for count, frame in enumerate(frames):
                plt.annotate(frame,(embedded[count,0], embedded[count,1]), fontsize = 6)
                ''' note: this annotation is for the frames in the VIDEO, NOT indices of the centroids / orientations! 
                Remember to subtract by shift '''

    def compute_speed_from_centroids(self):
        centroids = self.centroids
        speeds = np.zeros(self.frames-1)

        for i in range(1,self.frames):
            euc_dist = norm(centroids[i] - centroids[i-1])
            speed_i = euc_dist * 30 # unit: euclidean distance / second
            speeds[i - 1] = speed_i

        self.speeds = speeds

    def compute_horizontal_length(self):
        pass



class LED_Sync_batch():  # BETA. this class outputs the trials' start and end frames in the video
    # point of synchrony = first cue light as detected in function 'compute_first_trial_frame'
    def __init__(self, matfile, LED_file): # matfile = trial start times, LED_file = .npy file detected from video
        self.frame_rate = 30  # per second
        self.matfile_timestamp = sio.loadmat(matfile)['LED_timeDelay'] # loads trial start times from MATFILE
        self.matfile_frames = self.convert_to_frames(self.matfile_timestamp) # convert to frames according to frame rate
        self.LED_frames = np.sort(np.load(LED_file)) # flash timestamps (in FRAMES) computed directly from video
        self.title = matfile

    def convert_to_frames(self, timestamps):  # convert matfile timestamps to video frames
        frames = timestamps * self.frame_rate
        frames = np.ceil(frames)
        return frames

    def start_align(self, trial1_in_video_that_correspond_to_matfile = 1):

        # compare flash times from video, and start times from mat file
        LED_frames = self.LED_frames
        matfile_timestamp = self.matfile_timestamp[:,0]
        LED_1_trial, LED_end_trial = LED_frames[trial1_in_video_that_correspond_to_matfile - 1], LED_frames[-1]
        d_frames = LED_end_trial - LED_1_trial # delta frame computed using LED data from video capture
        d_time = matfile_timestamp[-1] - matfile_timestamp[0] # delta time in seconds computed from matfile
        callibrated_frame_rate = d_frames / d_time
        self.frame_rate = callibrated_frame_rate  # update frame rate
        self.matfile_frames = self.convert_to_frames(matfile_timestamp)  # re-map onto frames with old matfile timestamps
        self.frame_shift = self.matfile_frames[-1] - LED_end_trial  # compare last trial
        self.trial_1_LED = trial1_in_video_that_correspond_to_matfile-1

    def comparison_plot(self, matplotlib = True):
        LED_x = self.LED_frames[self.trial_1_LED:]
        LED_y = 11 * np.ones(LED_x.size)
        mat_x = self.matfile_frames - self.frame_shift
        mat_y = 10.5 * np.ones(mat_x.size)
        if matplotlib:
            plt.figure(figsize = (12,1))
            plt.scatter(LED_x, LED_y, color='blue', s=5)
            plt.scatter(mat_x, mat_y, color='red', s=5)
            plt.axis('off')
        else:
            data = []
            LED_trace = Scatter(x=LED_x, y=LED_y, mode = 'markers')
            data.append(LED_trace)
            mat_trace = Scatter(x=mat_x, y=mat_y, mode = 'markers')
            data.append(mat_trace)
            layout = Layout(
                title=self.title,
                xaxis=dict(title='Frames'),
                yaxis=dict(
                    range=[0, 20],
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False
                ))
            py.plot(Figure(data=data, layout=layout), filename='test')

# class batch_LED_matchfiles():
#     # this class is only for matching videos with correct matfiles, by comparing trial type and timestamps
#     def __init__(self, matfiles_dir, LEDfiles_dir):



class LED_Sync_single():  # this class outputs the trials' start and end frames in the video
    # point of synchrony = first cue light as detected in function 'compute_first_trial_frame'
    def __init__(self, matfile, LED_file): # matfile = trial start times, LED_file = .npy file detected from video
        self.frame_rate = 30  # per second
        self.matfile_timestamp = sio.loadmat(matfile)['LED_timeDelay'] # loads trial start times from MATFILE
        self.matfile_STIM_delay = sio.loadmat(matfile)['STIM_time']
        self.trial_TYPE = sio.loadmat(matfile)['TRIAL_TYPES']
        self.trial_TYPE = self.trial_TYPE.reshape(self.trial_TYPE.size)
        # pdb.set_trace()
        self.matfile_frames = self.convert_to_frames(self.matfile_timestamp) # convert to frames according to frame rate
        self.LED_frames = np.sort(np.load(LED_file)) # flash timestamps (in FRAMES) computed directly from video
        self.title = matfile

    def convert_to_frames(self, timestamps):  # convert matfile timestamps to video frames
        frames = timestamps * self.frame_rate
        frames = np.ceil(frames)
        return frames

    def start_align(self, trial1_in_video_that_correspond_to_matfile = -1): # if shift > 0, trim from LED, if <0, trim from matfile

        # TRUNCATE TRIALS THAT IS/ARE CUT FROM THE VIDEO AND MATCH FIRST TRIAL TO MATFILE
        if trial1_in_video_that_correspond_to_matfile >= 0:
            self.LED_frames = self.LED_frames[trial1_in_video_that_correspond_to_matfile:]
        else:
            self.matfile_timestamp = self.matfile_timestamp[-trial1_in_video_that_correspond_to_matfile:]
            self.matfile_frames = self.matfile_frames[-trial1_in_video_that_correspond_to_matfile:]
            self.matfile_STIM_delay = self.matfile_STIM_delay[-trial1_in_video_that_correspond_to_matfile:]
            self.trial_TYPE = self.trial_TYPE[-trial1_in_video_that_correspond_to_matfile:]
        LED_frames = self.LED_frames
        # matfile_timestamp = self.matfile_timestamp[:,1] # can decide to average or take 1st/last time steps
        matfile_timestamp = np.average(self.matfile_timestamp,1) # can decide to average or take 1st/last time steps
        d_frames = LED_frames[-1] - LED_frames[0] # delta frame computed using LED data from video capture
        d_time = matfile_timestamp[-1] - matfile_timestamp[0] # delta time in seconds computed from matfile
        callibrated_frame_rate = d_frames / d_time
        self.frame_rate = callibrated_frame_rate  # update frame rate
        self.matfile_frames = self.convert_to_frames(matfile_timestamp)  # re-map onto frames with old matfile timestamps
        self.frame_shift = self.matfile_frames[-1] - LED_frames[-1]  # compare last trial

    def align_STIM_times(self):
        '''
        this function can only be called after alignment is completed using LED TIMES.
        '''
        # STIM_delay_avg = np.average(self.matfile_STIM_delay,1)
        STIM_delay_end = self.matfile_STIM_delay[:,-1]
        STIM_delay_frames = STIM_delay_end * self.frame_rate
        self.STIM_aligned_frames = self.matfile_frames - self.frame_shift + STIM_delay_frames
        self.STIM_aligned_frames = self.STIM_aligned_frames.astype(int) # instead of rounding take integer!

    def comparison_plot(self, matplotlib = True):
        LED_x = self.LED_frames
        LED_y = 11 * np.ones(LED_x.size)
        mat_x = self.matfile_frames - self.frame_shift
        mat_y = 10.5 * np.ones(mat_x.size)
        if matplotlib:
            plt.figure(figsize = (12,1))
            plt.scatter(LED_x, LED_y, color='blue', s=5)
            plt.scatter(mat_x, mat_y, color='red', s=5)
            plt.axis('off')
        else:
            data = []
            LED_trace = Scatter(x=LED_x, y=LED_y,
                                mode = 'markers')
            data.append(LED_trace)
            mat_trace = Scatter(x=mat_x, y=mat_y,
                                mode = 'markers')
            data.append(mat_trace)
            layout = Layout(
                title=self.title,
                xaxis=dict(title='Frames'),
                yaxis=dict(
                    range=[0, 20],
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False
                ))
            py.plot(Figure(data=data, layout=layout), filename='test')

    def save_adjusted_matfile_frames(self):
        '''
        this function saves the combined output with:
        1. LED aligned FRAMES
        2. Stimuli aligned frames
        3. Trial types
        :return:
        '''
        LED_frames = self.matfile_frames - self.frame_shift

        combined_output = [LED_frames, self.STIM_aligned_frames, self.trial_TYPE]

        np.save(self.title + '_synced', combined_output)
        print(self.title + ' saved.')


class LED_Sync_old(): # this class outputs the trials' start and end frames in the video
    # point of synchrony = first cue light as detected in function 'compute_first_trial_frame'
    def __init__(self, file):
        self.LED_timestamps = sio.loadmat(file)['LED_timeDelay']
        self.frame_rate = 30 # per second
        self.LED_frames = self.convert_to_frames(self.LED_timestamps)

    def convert_to_frames(self, timestamps): # convert matfile timestamps to video frames
        frames = timestamps * self.frame_rate
        frames = np.ceil(frames)
        return frames

    def comparison_plots(self, Video, shift):
        plt.figure(figsize = (12,1.5))
        plt.plot(Video.all_LED_lum)
        shifted_LED_frames = self.LED_frames + shift
        plt.scatter(shifted_LED_frames, 56*np.ones(self.LED_frames.size), color = 'red', s = 2)
        
    def comparison_plot(self, LED_lum, title, shift = 0):
        '''
        plots all LED luminance (not detection) against the actual timestamps from the mat files,
        then plot it all on pylot (login to see changes)
        '''
        data = []
        max_LED_lum = max(LED_lum)
        LED_frames_zeroed = self.LED_frames - self.LED_frames[0,0]
        for i in range(LED_frames_zeroed.shape[0]):
            trace = Scatter(x = LED_frames_zeroed[i] + shift,
                            y = [max_LED_lum+1, max_LED_lum+1],
                            line = dict(color = ('rgb(205, 12, 24)')))
            data.append(trace)
        trace = Scatter(y=LED_lum, line = dict(color = 'black'))
        data.append(trace)
        layout = Layout(
            title=title,
            xaxis=dict(title='Frames'),
            yaxis=dict(title='Average luminance'))
        py.iplot(Figure(data=data, layout=layout), filename=title)

    def extract_flashframes_from_luminance(self, LED_lum):
        threshold = 150 # this threshold MUST be high. Otherwise, cannot distinguish b/w 2 consecutive lights
        LED_flashes = []
        for i, lum_i in enumerate(LED_lum):
            if lum_i > threshold:
                LED_flashes.append(i)
        # Group these into trials
        LED_flashes_bytrial = split(LED_flashes, where(diff(LED_flashes)>1)[0]+1)
        return LED_flashes_bytrial

    def align_LED_video_with_matfile(self, LED_lum, trial1_in_video_that_correspond_to_matfile):
        LED_flashes_bytrial = self.extract_flashframes_from_luminance(LED_lum)
        first_trial, last_trial = LED_flashes_bytrial[0][0], LED_flashes_bytrial[-1][0]
        diff_frames = last_trial - first_trial
        diff_times = self.LED_timestamps[-1, 0] - self.LED_timestamps[trial1_in_video_that_correspond_to_matfile-1, 0]
        callibrated_frame_rate = diff_frames / diff_times
        self.frame_rate = callibrated_frame_rate  # update frame rate
        self.LED_frames = self.convert_to_frames(self.LED_timestamps) # recompute frames from timestamps
        self.shift = self.LED_frames[-1,0] - LED_flashes_bytrial[-1][0]


class Plot(): # all plot related functions

    def __init__(self, smoothing, Video):

        self.shift = smoothing.shift
        self.Video = Video

    def keypoints_with_confidence(self, keypoints, confidence):
        # this function needs SHIFT added
        for i in range(frames):
            x_i = keypoints[i, 0, 0, :]
            y_i = keypoints[i, 0, 1, :]
            con_score_i = confidence[i, 0]
            plt.ion()
            plt.axis((0, 640, 660, 0))

            plt.imshow(self.Video.capture_frame(i))  # plot video

            plt.scatter(x_i[0], y_i[0], cmap='winter', c=con_score_i[0], marker='$H$')  # head
            plt.scatter(x_i[1], y_i[1], cmap='winter', c=con_score_i[1], marker='$L$')  # left forelimb
            plt.scatter(x_i[2], y_i[2], cmap='winter', c=con_score_i[2], marker='$R$')  # right forelimb
            plt.scatter(x_i[3], y_i[3], cmap='winter', c=con_score_i[3], marker='$N$')  # neck
            plt.scatter(x_i[4], y_i[4], cmap='winter', c=con_score_i[4], marker='$L$')  # left hindlimb
            plt.scatter(x_i[5], y_i[5], cmap='winter', c=con_score_i[5], marker='$R$')  # right hindlimb
            plt.scatter(x_i[6], y_i[6], cmap='winter', c=con_score_i[6], marker='$T$')  # trail

            plt.pause(0.01)
            plt.clf()

    def keypoints_only(self, keypoints, start, duration):
        shift = self.shift
        for i in range(start,start+duration):
            x_i = keypoints[i, 0, :]
            y_i = keypoints[i, 1, :]
            plt.ion()
            plt.axis((0, 640, 660, 0))
            plt.title(i)
            plt.imshow(self.Video.capture_frame(i + shift))  # plot video frame

            plt.scatter(x_i[0], y_i[0], color='red', marker='$N$')  # nose
            plt.scatter(x_i[1], y_i[1], color='blue', marker='$L$')  # left forelimb
            plt.scatter(x_i[2], y_i[2], color='blue', marker='$R$')  # right forelimb
            plt.scatter(x_i[3], y_i[3], color='white', marker='$N$')  # neck
            plt.scatter(x_i[4], y_i[4], color='green', marker='$L$')  # left hindlimb
            plt.scatter(x_i[5], y_i[5], color='green', marker='$R$')  # right hindlimb
            plt.scatter(x_i[6], y_i[6], color='white', marker='$T$')  # tail

            plt.pause(0.001)
            plt.clf()

    def centroids_only(self, centroids):
        for i in range(frames):
            x_i = centroids[i, 0]
            y_i = centroids[i, 1]
            plt.ion()
            plt.axis((0, 640, 660, 0))
            plt.imshow(self.Video.capture_frame(i))  # plot video
            plt.scatter(x_i, y_i, color='red', marker='$*$')  # head

            plt.pause(0.01)
            plt.clf()

    def plot_all_centroids(self, centroids):
        plt.figure()
        plt.imshow(Video.capture_frame(frames))
        plt.axis((0, 640, 660, 0))
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'o', color = 'red', s = 0.15)

    def plot_only_nose(self, keypoints):
        plt.figure()
        plt.imshow(Video.capture_frame(frames))
        plt.axis((0, 640, 660, 0))
        plt.scatter(keypoints[:, 0, 0], keypoints[:, 1, 0], marker = 'o', color = 'red', s = 0.15)

    def plot_centroid_and_nose(self, centroids, keypoints, start, duration):
        shift = self.shift
        # only plot one figure mode
        if start.__class__ == int:
            plt.figure()
            for i in range(start,start+duration):
                x1 = centroids[i - shift, 0]
                y1 = centroids[i - shift, 1]
                x2 = keypoints[i - shift, 0, 0]
                y2 = keypoints[i - shift, 1, 0]
                plt.ion()
                plt.axis((0, 640, 660, 0))
                plt.title('Frame in video: '+str(i))
                plt.imshow(self.Video.capture_frame(i))  # plot video frame

                plt.plot(x1, y1, x2, y2, marker='o')

                plt.pause(0.001)
                plt.clf()
        else: # MULTI-VIDEO SIMULTANEOUS PLOTTING. Start = list
            # fig, axarr = plt.subplots(len(start))
            skip_frames = 3 # display 1 frame in place of every 3 frames
            box = math.sqrt(len(start))
            box = math.ceil(box)
            plt.figure(figsize = (11,11))
            gs1 = gridspec.GridSpec(box,box)
            gs1.update(wspace=0, hspace=0)
            for i in range(0, duration, skip_frames):
                plt.clf()
                for j in range(len(start)):
                    ax1 = plt.subplot(gs1[j])
                    plt.axis('off')
                    frame = start[j]+i
                    ax1.imshow(self.Video.capture_frame(frame))
                    x1 = centroids[start[j] + i - shift, 0]
                    y1 = centroids[start[j] + i - shift, 1]
                    x2 = keypoints[start[j] + i - shift, 0, 0]
                    y2 = keypoints[start[j] + i - shift, 1, 0]
                    plt.plot(x1, y1, x2, y2, marker='o')
                    plt.annotate(start[j], (4, 45), fontsize=12, color = 'white')
                    plt.annotate(frame, (4, 85), fontsize=9, color='white')
                plt.pause(0.0001)

    def animate_multi_view_beta(self, start, duration, centroids, keypoints, skip_frames = 1,
                           interval = 1000/30, repeat = False, save = False, only_keypoints = False,
                                kp_confidence = None, smoothed = True, bounding_box = None,
                                plot_bbox = False, label = True):
        shift = self.shift
        box = math.ceil(math.sqrt(len(start)))
        fig = plt.figure(figsize = (10,10))
        gs1 = gridspec.GridSpec(box,box)
        gs1.update(wspace=0, hspace=0)
        ims = []
        print()
        print('Rendering video...')
        for i in range(0, duration, skip_frames):
            ims_i = []
            for j in range(len(start)):
                # CHOOSE SUBPLOT
                ax = plt.subplot(gs1[j])
                ax.axis('off')
                # PLOT FRAME
                img_i_j = self.Video.capture_frame(start[j]+i)
                frame_i_j = ax.imshow(img_i_j)
                # LABELS
                if label:
                    label_start_frame = ax.annotate(start[j], (4, 45), fontsize=12, color = 'white')
                    label_curr_frame = ax.annotate(start[j]+i, (4, 85), fontsize=9, color='white')

                if only_keypoints:
                    # PLOT ALL 7 KEYPOINTS
                    if smoothed:
                        x_i = keypoints[start[j] + i - shift, 0, :]
                        y_i = keypoints[start[j] + i - shift, 1, :]
                        plot_kpoints = ax.scatter(x_i, y_i, c='cyan', s = 150)
                    else:
                        x_i = keypoints[start[j] + i, 0, 0, :]
                        y_i = keypoints[start[j] + i, 0, 1, :]
                        con_score_i = kp_confidence[i, 0]
                        plot_kpoints = ax.scatter(x_i, y_i, cmap = plt.cm.coolwarm, c = con_score_i,
                                                  s = 150)
                    # COMBINE ALL ACTIONS
                    if label:
                        ims_i.extend((frame_i_j, plot_kpoints, label_start_frame, label_curr_frame))
                    else:
                        ims_i.extend((frame_i_j, plot_kpoints))
                else:
                    # PLOT CENTROID / NOSE
                    x1, y1 = centroids[start[j] + i - shift, 0], centroids[start[j] + i - shift, 1]
                    x2, y2 = keypoints[start[j] + i - shift, 0, 0], keypoints[start[j] + i - shift, 1, 0]
                    line, = ax.plot([x1, x2], [y1, y2], c = 'C0', linewidth = 6)
                    centroid, = ax.plot(x1, y1, marker='o', c='cyan', ms = 30)
                    nose, = ax.plot(x2, y2, marker='o', c='lime', ms = 30)
                    # COMBINE ALL ACTIONS
                    if label:
                        ims_i.extend((frame_i_j, line, centroid, nose, label_start_frame, label_curr_frame))
                    else:
                        ims_i.extend((frame_i_j, line, centroid, nose))
                ####
                # if plot_bbox:
                #     x1, y1 = bounding_box[start[j] + i, 0], bounding_box[start[j] + i, 1]
                #     x2, y2 = bounding_box[start[j] + i, 2], bounding_box[start[j] + i, 3]
                #     w = x2 - x1
                #     h = y2 - y1
                #     rect = patches.Rectangle((x1, y1), w, h,linewidth=1,edgecolor='r',facecolor='none')
                #     bbox = ax.add_patch(rect)
                #     ims_i.append(bbox)
                # ####
            ims.append(ims_i)
        Anim = animation.ArtistAnimation(fig, ims, interval = interval, blit = True, repeat = repeat, repeat_delay=3000)
        if save:
            if only_keypoints:
                Anim.save(str(start) + '_' + str(duration) + '_kps.mp4')
            else:
                Anim.save(str(start) + '_' + str(duration) + '_centroids.mp4')
        return Anim

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

