from copyreg import pickle
from doctest import run_docstring_examples
from importlib.util import module_for_loader
import pandas as pd
from os import path
from bids.layout import BIDSLayout

import warnings
warnings.filterwarnings('ignore')

def model_segmented(run_img, sub, ses_num, run_num, task_name, run_events, condf, fwd, recorded_tr=0.610, brain_mask='/foundcog/templates/mask/nihpd_asym_02-05_mask_2mm.nii.gz', fwd_cutoff=0.5):    
    outpaths = []
    if 'vid' not in task_name:
        return outpaths
    
    ### FUNCTIONS DEFINED WITHIN MODULE BECAUSE OF NIPYPE ##
    def _cols_to_remove(desmat):
        dropcols = [j for j in [i for i in list(desmat.columns) if not 'reg' in i] if not 'drift' in j]
        dropcols.remove('constant')
        return dropcols
    
    def _first_frame_idx(events, key, t_r):
        key_onset = events[events['trial_type'].str.contains(key)].onset.values
        key_idx = np.round(key_onset/t_r)
        return int(key_idx)
    
    ## IMPORTS WITHIN BECAUSE OF NIPYPE ##
    import os
    import pickle
    import pandas as pd
    import numpy as np

    from nilearn.image import get_data
    from nilearn._utils.niimg_conversions import check_niimg
    from nilearn.glm.first_level import FirstLevelModel ,make_first_level_design_matrix

    all_orders = {
        'A': ['minions_supermarket.mp4','new_orleans.mp4','bathsong.mp4','dog.mp4','moana.mp4','forest.mp4'], 
        'B': ['bathsong.mp4','dog.mp4','moana.mp4','forest.mp4','minions_supermarket.mp4','new_orleans.mp4'], 
        'C': ['new_orleans.mp4','minions_supermarket.mp4','dog.mp4','bathsong.mp4','forest.mp4','moana.mp4'], 
        'D': ['moana.mp4','forest.mp4','minions_supermarket.mp4','new_orleans.mp4','bathsong.mp4','dog.mp4'], 
        'E': ['forest.mp4','moana.mp4','new_orleans.mp4','minions_supermarket.mp4','dog.mp4','bathsong.mp4'], 
        'F': ['dog.mp4','bathsong.mp4','forest.mp4','moana.mp4','new_orleans.mp4','minions_supermarket.mp4']
    }
    
    run_events = pd.read_csv(run_events,sep='\t')
    condf = pd.read_csv(condf, header=None, sep='  ')
    fwd = pd.read_csv(fwd)
    
    # check if there's no full order
    if len(run_events) < 7:
        print(f'no full order for sub-{sub} ses {ses_num} run {run_num}')
        return outpaths
    
    # check if there's at least one full order
    if len(run_events) >= 7 and len(run_events) < 14:
        one_order = True
    else:
        one_order = False
    
    # initialise the model 
    model = FirstLevelModel(t_r=recorded_tr, minimize_memory=False, mask_img=brain_mask)

    # get which orders for this participant
    sub_orders = []
    for ag_idx in run_events.index[run_events['trial_type'].str.contains('attention_getter')]:
        _firstvid = run_events['trial_type'][ag_idx+1]
        _find=True
        while _find==True:
            for k,v in all_orders.items():
                if v[0]==_firstvid:
                    sub_orders.append(k)
                    _find=False

    run_events['trial_type'] = run_events['trial_type'].str.replace('.mp4','')
    # tag each trial type with a label to split design correctly
    _runtags = ['ag1','vid1a','vid2a','vid3a','vid4a','vid5a','vid6a','ag2','vid1b','vid2b','vid3b','vid4b','vid5b','vid6b']
    _tagged_tts = []
    for idx,tt in enumerate(run_events.trial_type):
        _tagged_tts.append(f'{_runtags[idx]}_{tt}')
    run_events['trial_type'] = _tagged_tts
    
    # BUILD FRAME TIMES - code snippet taken from nilearn
    n_scans = get_data(run_img).shape[3]
    start_time = model.slice_time_ref * model.t_r
    end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
    frame_times = np.linspace(start_time, end_time, n_scans)

    # Get indices where this is over cutoff
    # #   Be careful of whether setting 1st or 2nd scan of difference
    # THIS IS FIRST SCAN
    above_idxs = fwd.index[fwd['FramewiseDisplacement']>fwd_cutoff].values
    
    # THIS WOULD BE SECOND SCAN
    # above_idxs = above_idxs + 1

    # 3. Construct matrix with nrows=nscans, ncols=nframes_todrop
    spike_arr = np.zeros((len(fwd)+1,above_idxs.size))
    spike_arr[above_idxs,np.arange(above_idxs.size)] = 1

    # Add spike regressors along axis 1 of condf array
    confounds = np.concatenate((condf,spike_arr),axis=1)
    confounds = np.nan_to_num(confounds)

    # Check to see if number of spikes is over threshold
    discard_thresh = 0.5
    if len(above_idxs) > len(fwd) * discard_thresh:
        print(f'Too much motion in sub-{sub} ses {ses_num} run {run_num}, skipping')
        return outpaths
    
    # MAKE DESIGN MATRIX
    design = make_first_level_design_matrix(    frame_times,
                                                events=run_events,
                                                hrf_model=model.hrf_model,
                                                drift_model=model.drift_model,
                                                high_pass=model.high_pass,
                                                drift_order=model.drift_order,
                                                fir_delays=model.fir_delays,
                                                add_regs=confounds,
                                                min_onset=model.min_onset
                                                )
    # find the indices of vid1a , ag2 , vid1b
    # Use these to split the design matrix and scan, as attention getter too mischievious
    _vid1a = _first_frame_idx(run_events, 'vid1a', model.t_r)

    # we want to get the same length segments
    # also should include up to 10 s at end of video
    # the max length should be somewhere around 224 for the first segment, then second varies depending on end of scan but usually >240
    length_of_segs = 234

    # split the design matrix into its two halves
    # make sure to use iloc not loc or normal slicing as the dataframe indices are the frame times
    design_one = design.iloc[_vid1a:_vid1a+length_of_segs]
    
    # almost ready to fit, first we need to remove the regressors of interest
    # this model is for the ISC analysis, with the purpose of denoising so we only want to model the confound regressors here, ensuring we save the residuals
    design_one = design_one.drop(_cols_to_remove(design_one), axis=1)
    
    # repeat for case where there's a second order
    if not one_order:
        _vid1b = _first_frame_idx(run_events, 'vid1b', model.t_r)
        # check first for case where events are much longer than fMRI, and second order should be discounted
        if _vid1b > len(design):
            one_order = True
        # then, is the desired length longer than actual length - use actual if so
        elif _vid1b+length_of_segs > len(design):
            design_two = design.iloc[_vid1b:]    
            design_two = design_two.drop(_cols_to_remove(design_two), axis=1)
        # finally use the desired length, if we have enough data to do so
        else:
            design_two = design.iloc[_vid1b:_vid1b+length_of_segs]
            design_two = design_two.drop(_cols_to_remove(design_two), axis=1)
    
    # get rid of empty columns that had spikes but now the frame is not in our segment
    design_one = design_one.loc[:, (design_one != 0).any(axis=0)]
    if not one_order:
        design_two = design_two.loc[:, (design_two != 0).any(axis=0)]

    fit=True
    if fit:
        # split the scan into corresponding lengths
        run_img = check_niimg(run_img, ensure_ndim=4)

        segment_one = run_img.slicer[:,:,:,_vid1a:_vid1a+len(design_one)]

        # finally, fit the model
        model.fit(segment_one, design_matrices=design_one)
        one_path = os.path.abspath(f'order-{sub_orders[0]}_sub-{sub}_ses-{ses_num}_run-{run_num}_confound-model.pickle')
        with open(one_path,'wb') as f:
            pickle.dump(model, f)
        # free up memory
        del model
        
        if not one_order:
            segment_two = run_img.slicer[:,:,:,_vid1b:_vid1b+len(design_two)]
            model_two = FirstLevelModel(t_r=recorded_tr, minimize_memory=False, mask_img=brain_mask)
            model_two.fit(segment_two, design_matrices=design_two)
            two_path = os.path.abspath(f'order-{sub_orders[1]}_sub-{sub}_ses-{ses_num}_run-{run_num}_confound-model.pickle')
            with open(two_path,'wb') as f:
                pickle.dump(model_two, f)
            # free up memory
            del model_two

            outpaths.append((one_path,two_path))
        else:
            outpaths.append((one_path,))
    return outpaths

if __name__ == '__main__':
    outpaths=model_segmented(
        '/foundcog/bids/workingdir/ICC103/derivatives/preproc/_subject_id_ICC103/_referencetype_standard/_run_002_session_2_task_name_videos/smoothing/sub-ICC103_ses-2_task-videos_dir-AP_run-002_bold_mcf_corrected_flirt_smooth.nii.gz',
        'ICC103',
        '2',
        '002',
        'videos',
        '/foundcog/bids/sub-ICC103/ses-2/func/sub-ICC103_ses-2_task-videos_dir-AP_run-002_events.tsv',
        '/foundcog/bids/workingdir/ICC103/derivatives/preproc/bold_preproc/_subject_id_ICC103/_run_002_session_2_task_name_videos/_referencetype_standard/mcflirt/sub-ICC103_ses-2_task-videos_dir-AP_run-002_bold_mcf.nii.par',
        '/foundcog/bids/workingdir/ICC103/derivatives/preproc/bold_preproc/_subject_id_ICC103/_run_002_session_2_task_name_videos/_referencetype_standard/calc_fwd/fd_power_2012.txt',
        0.61,
        '/foundcog/templates/mask/nihpd_asym_02-05_mask_2mm.nii.gz',
        1.5
    )