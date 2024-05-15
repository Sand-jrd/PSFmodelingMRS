#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:37:04 2024

@author: sand-jrd
"""


# File managment
from os import mkdir
from os.path import isdir
from astropy.io import fits
import pickle

# Misc
import numpy as np

# All function to do the job. One for each step.
from sandpipline import (get_all_rate_file_at_path, 
                         select_dither_band,
                         select_dither_type, 
                         select_dither_by_postion,
                         create_spline_cube,
                         re_scale_spline_frame,
                         re_scale_shift_spline_frame)

# Extra for plots
from sandpipline import (plot_dither_slice_by_postion, 
                         show_data_point,
                         flush_current_figs,
                         multipage)

# %%  Parameters

# Give directory where distortion cdps are located
datadir   = '../obs4_MRS_SHORT'
cdpDir   = '../mrs_distortion_fits/'
science_data_dir = "../jwst1050_rate"#'./GQLup/GQLupShort'# 


# %% Read files 


# --- Process refs, or read it if algready processed
have_a_save = True
ref_pickel_name = "../rate_ref"#+"_med"

if have_a_save :
    
    with open(ref_pickel_name, 'rb') as handle:
        saved_rate_infos = pickle.load(handle)
    cube_frame, centers_list, dith_type, numdi_list, name_list, bandCHAN = saved_rate_infos
else : 
    cube_frame, centers_list, dith_type, numdi_list, name_list, bandCHAN = get_all_rate_file_at_path(datadir, cdpDir, ref_pickel_name)

# --- Process Sceince data, or read it if already processed
have_a_save = True
sci_pickel_name = "../rate_sci"#+"GQLup"

if have_a_save :

    with open(sci_pickel_name, 'rb') as handle:
        saved_rate_infos = pickle.load(handle)
    cube_frame_sci, centers_list_sci, dith_type_sci, numdi_list_sci, name_list_sci, bandCHAN_sci = saved_rate_infos

else : 
    cube_frame_sci, centers_list_sci, dith_type_sci, numdi_list_sci, name_list_sci, bandCHAN_sci = get_all_rate_file_at_path(science_data_dir, cdpDir ,sci_pickel_name )

# %% Process the files
    
# Save in a new folder
save_dir  = science_data_dir+'_subtracted/' # Folder for the processed science
if not isdir(save_dir): mkdir(save_dir)

save_dir2   = science_data_dir+'_MODEL/' # Folder for the model
if not isdir(save_dir2): mkdir(save_dir2)

for kk in range(len(cube_frame_sci)):
    
    flush_current_figs()
    
    # We select one frame from the list of science rate file to process. 
    frame_sci = cube_frame_sci[kk]
    type_sci = dith_type_sci[kk]
    centers_sci = centers_list_sci[kk]
    name_rate = name_list_sci[kk]
    bandsci = bandCHAN_sci[kk]

    # -- Select the dither from the correct band.
    cube_frame_band_i, centers_band_i, numdi_band_type_i, dith_type_band_i = select_dither_band(bandsci, 
                                                                            cube_frame, 
                                                                            centers_list, 
                                                                            dith_type, 
                                                                            numdi_list,
                                                                            bandCHAN)

    # -- Select the dither from the right type.
    cube_frame_type_i, centers_type_i, numdi_list_type_i = select_dither_type(type_sci, 
                                                                            cube_frame_band_i, 
                                                                            centers_band_i, 
                                                                            dith_type_band_i, 
                                                                            numdi_band_type_i)
    
    # -- From this selection, lets sort them according to the beta position now.
    cube_frame_3diths, centers_3diths = select_dither_by_postion(cube_frame_type_i, 
                                                                centers_type_i, 
                                                                numdi_list_type_i, 
                                                                centers_sci, 
                                                                show=False)
    
    # PS : You can also plot the slice for each row od dither with :
    # plot_dither_slice_by_postion(cube_frame_type_i, centers_type_i)
    # show_data_point(np.array([frame_sci]),np.array([centers_sci]),numdi_list="Science", chose_1color="tab:red")
    # show_data_point(cube_frame_3diths, centers_3diths,numdi_list="References", chose_1color="tab:blue")

    # -- Now we can use this frame, with assosiated alpha coordinate to creat the spline model.
    # This fuction can return the list of spline fuction, or the cube of interpolated spline, it the science alpha coordinate is provided.
    cube_spline, list_spline_funct = create_spline_cube(cube_frame_3diths, 
                                                        centers_3diths, 
                                                        centers_sci,
                                                        s=2, k=4, show_slicey=True, frame_sci=frame_sci)
    
    ### -- Rescale each row one by one to fit the data
    #model_frame = re_scale_spline_frame(cube_spline, frame_sci, fitting_window_left=range(100,500), fitting_window_right=range(80,250),  show=True)
    model_frame = re_scale_shift_spline_frame(list_spline_funct, frame_sci, fitting_window_left=range(100,500), fitting_window_right=range(40,400),  show=True)
    residual = np.log10(frame_sci) - model_frame


    residual = np.log10(frame_sci) - model_frame
    multipage("results"+str(kk)+".pdf")

    #  --Load original file, re-write in in a new folder with updating data to residual
    # This is to we keep the headers.
    hdul = fits.open(science_data_dir+"/"+name_rate)
    hdul.verify('ignore')
    hdul[1].data = 10**(residual)-1
    hdul.writeto(save_dir+name_rate, overwrite=True)

    hdul = fits.open(science_data_dir+"/"+name_rate)
    hdul.verify('ignore')
    hdul[1].data = 10**(model_frame)-1
    hdul.writeto(save_dir2+name_rate, overwrite=True)
    
    
    
