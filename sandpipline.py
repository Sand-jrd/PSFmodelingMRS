#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:22:47 2024

@author: sand-jrd


This file contain functions for each step of the creation of Spline fittinf : 
        
    1 - get_all_rate_file_at_path : Read the rates from a file and compute the centorids.
        Values are stored in a big list that have to be pass to next functions.
        The 2 next functions will help to selected the appropriate files from the list
    
    2 - select_dither_type : Create sub selection of dither list of one specific type : 
        NEGATIVE1', 'NEGATIVE2', 'POSTITIVE'... 
    
    
    3 - select_dither_by_postion : Create sub selection of dither list, based on beta coordinate
        Will compare beta position with science data and take the 3 closed dither point.
    
    prep.4 - set_weights &  remove_outliers
   
    4 - create_spline_cube : Create the spline model from the dither list provided.
    
    5 - re_scale_spline_frame : Reslace the model to fit at best the science, 
        slice by slice, and for both left and right side individually
    
    5bis - re_scale_slice : Sub-function used by (5), to rescale one slice given fitting window

Finally, few function at the end aim at assesing the point cloud quality : 
v    
    1 - Print the model relative to the cloud point
    
    Nearest neighbor analysis: Compute the disparity of nerby points 
    2 - Interpolation error : Error and svd between Spline model and data-point.
        Can be apply both compare to references and science data points.

"""

# # Misc
import numpy as np
import pickle
import matplotlib.pyplot as plt

# log10 raises warning when ther is a negative value.
# It return Nan's for thoses points and that is what we want
np.seterr(invalid='ignore') # We do no want the warning tho, it is enoying


a_to_pix = 0.196 # Value used to convert alpha shift into pixel displacement
band_id = {"LONG":"C", "MEDIUM":"B", "SHORT":"A"} # Convert band name into letter

# Plot utils.
colors_type = {'NEGATIVE1':"tab:orange", 'NEGATIVE2':"tab:blue", 'NEGATIVE3':"tab:green", 'NEGATIVE4':"tab:purple",
          'POSITIVE1':"tab:red",'POSITIVE2':"blue", 'POSITIVE3':"tab:pink", 'POSITIVE4':"pink"}

colors = ["tab:orange","tab:blue","tab:green","tab:purple","tab:red","blue", "tab:pink", "pink", "orange", "tab:orange","tab:blue","tab:green","tab:purple","tab:red","blue", "tab:pink", "pink", "orange"]

# %% Two little function to generate a pdf pages containing useful plot

def flush_current_figs():
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    
    plt.clf()
    plt.cla()
    plt.close("all")


def multipage(filename, figs=None, dpi=200):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

# %% 

def get_spec_grid(d2cMaps):
    
    # spec_grid_left
    lambmin = np.nanmin(d2cMaps['lambdaMap'][np.where(d2cMaps['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps['lambdaMap']) # micron
    lambcens = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    lambfwhms = np.ones(len(lambcens))*(2*(lambmax-lambmin)/1024.)    
    
    return [lambcens,lambfwhms]


def get_all_rate_file_at_path(path, cdpDir, save_pickle=None, bkg=None, plot_and_save=True):
    
    # File managment
    from glob import glob
    from astropy.io import fits
    
    # To get centroids
    from distortionMaps import d2cMapping
    from funcs_sand import point_source_centroiding

    # To get bad pixels flags
    from jwst.datamodels import dqflags
    
    
    cube_frame =[]
    dith_type = []
    numdi_list = []
    name_list = []
    centers = []
    bandCHAN  = []
    
    if not glob(path+"/*_rate.fits") :
        raise FileNotFoundError("No rate files found matching path : "+path+"/*_rate.fits")

    if bkg:
        bkg_hdul=[]
        for bkgfile in glob(bkg+"/*_rate.fits"):
            hdul = fits.open(bkgfile)
            bkg_hdul.append(hdul)

    for file in glob(path+"/*_rate.fits"):
        
    
        name =  file.split('/')[-1]
        par = name[24]
        numdi = name[14:19]
        
    
        ### -- load fits file and get infos
        hdul = fits.open(file)
        hdul.verify('ignore')
        frame = (hdul[1].data)
        dq = hdul[3].data
        
        # Read the band from the header
        band = band_id[hdul[0].header["BAND"]]
        channel = hdul[0].header["CHANNEL"]
        band_left = channel[0]+band
        band_right = channel[1]+band
        
        # Create the mapping FOR BOTH CHANNEL
        d2cMaps_left = d2cMapping(band_left,cdpDir,slice_transmission='80pc',fileversion = "flt5")
        d2cMaps_right = d2cMapping(band_right,cdpDir,slice_transmission='80pc',fileversion = "flt5")


        ### -- Remove nan and bad pixels : 

        DO_NOT_USE = dqflags.pixel['DO_NOT_USE']
        NON_SCIENCE = dqflags.pixel['NON_SCIENCE']
        
        mask_bad = (np.bitwise_and(dq, DO_NOT_USE) == DO_NOT_USE)
        mask_non_science = (np.bitwise_and(dq, NON_SCIENCE) == NON_SCIENCE)
    
        frame[np.where(mask_bad+mask_non_science)]=np.nan
                
        if bkg : 
            found_bkg=True
            for bkg_i in bkg_hdul:
                if bkg_hdul[0].header["BAND"] == band and \
                   bkg_hdul[0].header["CHANNEL"]  == channel:
                       frame = frame-bkg_hdul[3].data
            if not found_bkg :    
                print("Not matching bkg. -- No backround subtraction on "+file)
                    
        
        # Append frame
        cube_frame.append( frame )
    
    
        ### --  Get centroid 
        
        # Left centorids
        centroid_2d = point_source_centroiding(band_left,frame,d2cMaps_left,spec_grid=get_spec_grid(d2cMaps_left), fit='2D')
        _ ,alpha_centers_l,beta_centers_l,_ ,_ ,_  = centroid_2d
        
        shiftx_l = np.mean(alpha_centers_l[~np.isnan(alpha_centers_l)])/a_to_pix
        shifty_l = np.mean(beta_centers_l[~np.isnan(beta_centers_l)])/a_to_pix

        #---- I update the point_source_centroiding, so if the gauss fit fail it return the argmax.
        # If you are using the version that can return Nan, then uncomment these lines. (same for right centrodds)
        # if np.isnan(shiftx_l): shiftx_l=0 # If shifty_l is empty replace with 0
        # if np.isnan(shifty_l): shifty_l=0

        # Right centorids
        centroid_2d = point_source_centroiding(band_right,frame,d2cMaps_right,spec_grid=get_spec_grid(d2cMaps_right), fit='2D')
        _ ,alpha_centers_r,beta_centers_r,_ ,_ ,_  = centroid_2d
        
        shiftx_r = np.mean(alpha_centers_r[~np.isnan(alpha_centers_r)])/a_to_pix
        shifty_r = np.mean(beta_centers_r[~np.isnan(beta_centers_r)])/a_to_pix
    
        # if np.isnan(shiftx_r): shiftx_r=0 # If shifty_l is empty replace with 0
        # if np.isnan(shifty_r): shifty_r=0

    
        # Store it in 2x2 tuple
        centers.append( ((shiftx_l,shifty_l), 
                          (-shiftx_r,-shifty_r)) )
        
        
        ## -- Append other file infos to be return
        
        direction = hdul[0].header["DITHDIRC"]
        dith_type.append(direction+par)    
        numdi_list.append(numdi)
        name_list.append(name)
        bandCHAN.append(band+hdul[0].header["CHANNEL"])
        
        if plot_and_save:
            
            plt.figure(file)
            sliceI= np.log10(1+frame[512])
        
            sliceI_l = sliceI[:512]
            max_gauss_l = np.nanargmax(sliceI_l)
            maxmin= [-1,np.nanmax(1.2*sliceI_l)]
            
            start_slice = max_gauss_l-20 + np.nanargmin(d2cMaps_left["alphaMap"][512][max_gauss_l-20:max_gauss_l])
            end_slice = max_gauss_l +1+ np.nanargmax(d2cMaps_left["alphaMap"][512][max_gauss_l:max_gauss_l+20])

            max_gauss_l_alpha = start_slice+np.nanargmin(abs(d2cMaps_left["alphaMap"][512][start_slice:end_slice]))

            plt.subplot(221)
            plt.title("Estimated centroïd - left")
            plt.xlim([max_gauss_l_alpha-10, max_gauss_l_alpha+10])
            plt.ylim(maxmin)
            plt.plot(sliceI_l, color="tab:orange")
            plt.plot([max_gauss_l_alpha+shiftx_l, max_gauss_l_alpha+shiftx_l], maxmin, color="tab:blue")
            plt.fill_between([max_gauss_l_alpha+shiftx_l-0.5, max_gauss_l_alpha+shiftx_l+0.5],[maxmin[0],maxmin[0]],[maxmin[1],maxmin[1]],alpha=0.5,color="tab:blue", label="centoïd +10.5pix")

            plt.plot(sliceI, label="Slice 512")
            
            plt.subplot(222)
            sliceI_r = sliceI[512:]
            max_gauss_r = 512+np.nanargmax(sliceI_r)
            maxmin= [-1,np.nanmax(1.2*sliceI_r)]
            
            start_slice = max_gauss_r-20 + np.nanargmax(d2cMaps_right["alphaMap"][512][max_gauss_r-20:max_gauss_r])
            end_slice = max_gauss_r +1+ np.nanargmin(d2cMaps_right["alphaMap"][512][max_gauss_r:max_gauss_r+20])

            max_gauss_r_alpha = start_slice+np.nanargmin(abs(d2cMaps_right["alphaMap"][512][start_slice:end_slice]))

            plt.title("Estimated centroïd - left")
            plt.xlim([max_gauss_r_alpha-10, max_gauss_r_alpha+10])
            plt.ylim(maxmin)
            plt.plot(sliceI, color="tab:orange")
            #plt.plot([max_gauss_r_alpha+shiftx_r, max_gauss_r_alpha+shiftx_r], maxmin, color="tab:blue", label="Applaying + shiftx")
            #plt.plot([max_gauss_r_alpha, max_gauss_r_alpha], maxmin, color="tab:red", label="AlphaMAP 0")
            plt.plot([max_gauss_r_alpha-shiftx_r, max_gauss_r_alpha-shiftx_r], maxmin, color="tab:green", label="Applaying - shiftx")
            #plt.fill_between([max_gauss_r_alpha+shiftx_r-0.5, max_gauss_r_alpha+shiftx_r+0.5],[maxmin[0],maxmin[0]],[maxmin[1],maxmin[1]],alpha=0.5, color="tab:blue")
            plt.fill_between([max_gauss_r_alpha-shiftx_r-0.5, max_gauss_r_alpha-shiftx_r+0.5],[maxmin[0],maxmin[0]],[maxmin[1],maxmin[1]],alpha=0.5, color="tab:green")
            plt.subplot(223)
            
            plt.title("Bin estimate of alpha_centers - Left")
            plt.plot(alpha_centers_l[~np.isnan(alpha_centers_l)]/a_to_pix)
            plt.plot([shiftx_l]*len(alpha_centers_l[~np.isnan(alpha_centers_l)]/a_to_pix))
            plt.ylim([shiftx_l+0.5, shiftx_l-0.5])

            plt.subplot(224)
            plt.title("Bin estimate of alpha_centers - Right")
            plt.plot(alpha_centers_r[~np.isnan(alpha_centers_r)]/a_to_pix)
            plt.plot([shiftx_r]*len(alpha_centers_r[~np.isnan(alpha_centers_r)]/a_to_pix))
            plt.ylim([shiftx_r+0.5, shiftx_r-0.5])


    dith_type = np.array(dith_type)

    multipage(path+".pdf")
    flush_current_figs()
    
    if save_pickle is not None: 
        with open(save_pickle, 'wb') as handle:
            pickle.dump((cube_frame, centers, 
                         dith_type, numdi_list, name_list, bandCHAN)
                        , handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return cube_frame, centers, dith_type, numdi_list, name_list, bandCHAN

# %%

def select_dither_type(type_i, cube_frame, centers, dith_type, numdi_list, best_row=False):
    """Select the list of dither of one specific type
    enuum_types = ['NEGATIVE1', 'NEGATIVE2', 'NEGATIVE3', 'NEGATIVE4',
                   'POSITIVE1','POSITIVE2','POSITIVE3', 'POSITIVE4']"""
   
    cube_frame_type_i = []
    centers_lists_type_i = []
    numdi_list_type_i = []
    
    # For each dither frame
    for kk in range(len(cube_frame)):

        # If the type match 
        if dith_type[kk]==type_i :
            
            # Add to the list 
            cube_frame_type_i.append(cube_frame[kk])
            centers_lists_type_i.append(centers[kk])
            numdi_list_type_i.append(numdi_list[kk])

    
    cube_frame_type_i = np.array(cube_frame_type_i)
    centers_lists_type_i = np.array(centers_lists_type_i)
    
    if not cube_frame_type_i.size :
        raise RuntimeError("No matching type, required type is : "+type_i)

    return cube_frame_type_i, centers_lists_type_i, numdi_list_type_i
 
    
def select_dither_band(band_sci, cube_frame, centers, dith_type, numdi_list, bandCHAN, best_row=False):
    """Select the list of dither of one specific band/chanel (A12, B12 ...)"""
   
    cube_frame_band_i = []
    centers_lists_band_i = []
    numdi_list_band_i = []
    dith_band_i = []
    # For each dither frame
    for kk in range(len(cube_frame)):

        # If the type match 
        if bandCHAN[kk]==band_sci :
            
            # Add to the list 
            cube_frame_band_i.append(cube_frame[kk])
            centers_lists_band_i.append(centers[kk])
            numdi_list_band_i.append(numdi_list[kk])
            dith_band_i.append(dith_type[kk])
    
    cube_frame_band_i = np.array(cube_frame_band_i)
    centers_lists_band_i = np.array(centers_lists_band_i)
    dith_band_i = np.array(dith_band_i)

    if not cube_frame_band_i.size :
        raise RuntimeError("No matching band, required type is : "+band_sci)

    return cube_frame_band_i, centers_lists_band_i, numdi_list_band_i, dith_band_i

# %%

def select_dither_by_postion(cube_frame, centers_lists, numdi_list, center_sci, show=False):
    """"Select the list of dither that matches best the science data"""
     

    # Get the position of the sci dither
    ((x_sci_l,y_sci_l), (x_sci_r,y_sci_r)) = center_sci
      
    # For each frame in the provided cube, check if compute beta distance
    beta_dist = []
    for kk in range(len(cube_frame)):
       
        (shiftx_l,shifty_l), (shiftx_r,shifty_r) = centers_lists[kk]

        beta_dist.append(abs(shifty_l-y_sci_l))
        
        if show:

            import matplotlib.pyplot as plt
            
            if kk==0 : plt.figure("Dither positions")
            plt.subplot(1,2,1)
            plt.title("Plot dither positions -- letf")
            plt.plot(-shiftx_l*a_to_pix, shifty_l*a_to_pix,"s" ,label=numdi_list[kk], markersize=20)  # Plot one frame.
            plt.annotate(numdi_list[kk][-1], (-shiftx_l*a_to_pix, shifty_l*a_to_pix), color="w", size=17, xytext=(-5, -5), textcoords='offset points')

            plt.xlabel("Alpha axis")
            plt.ylabel("Beta axis")
            
            plt.plot(-x_sci_l*a_to_pix, y_sci_l*a_to_pix,"s" , color="black", markersize=20)  # Plot one frame.
            plt.annotate("sci ", (-x_sci_l*a_to_pix, y_sci_l*a_to_pix), color="w", size=14, xytext=(-9, -5), textcoords='offset points')
            
            plt.subplot(1,2,2)

            plt.title("Plot dither positions -- right")
            plt.plot(-shiftx_r*a_to_pix, shifty_r*a_to_pix,"s" ,label=numdi_list[kk], markersize=20)  # Plot one frame.
            plt.annotate(numdi_list[kk][-1], (-shiftx_r*a_to_pix, shifty_r*a_to_pix), color="w", size=17, xytext=(-5, -5), textcoords='offset points')

            plt.xlabel("Alpha axis")
            plt.ylabel("Beta axis")

            plt.plot(-x_sci_r*a_to_pix, y_sci_r*a_to_pix,"s" , color="black", markersize=20)  # Plot one frame.
            plt.annotate("sci ", (-x_sci_r*a_to_pix, y_sci_r*a_to_pix), color="w", size=14, xytext=(-9, -5), textcoords='offset points')

    # Keep the 2 closest point.
    idc = np.argsort(beta_dist)

    idc_3best = idc[0:3]
    return cube_frame[idc_3best, :, :], centers_lists[idc_3best]

# %%

def remove_outliers(x_coors, y_coors, nb_neighb=15, std_ratio=2):
    """ Remove outliers from a set of points (x_coords, y_coords) based on their 
    standard deviation relative to the close neighborhood."""
    
    new_x_coors = []
    new_y_coors = []
    
    for ii in range(0, len(x_coors)-nb_neighb, nb_neighb):
        
        # Get the neighborhood
        neighborhood_x = x_coors[ii, ii+nb_neighb]
        neighborhood_y = y_coors[ii, ii+nb_neighb]
        
        # If the x-value of a point if above std_eatio threshold, discard
        good_neighbors = np.where(neighborhood_x < np.mean(neighborhood_x) + std_ratio*np.std(neighborhood_x)) *\
                         np.where(neighborhood_x > np.mean(neighborhood_x) - std_ratio*np.std(neighborhood_x))
        
        new_x_coors = np.concatenate(new_x_coors, neighborhood_x[good_neighbors])
        new_y_coors = np.concatenate(new_y_coors, neighborhood_y[good_neighbors])
        
    return new_x_coors, new_y_coors

def set_weights(x_coors, nb_neighb=20, std_ratio=3, show=False):
    """ Create weights array (size of x_coors) 
    1 - to deal with outliers based on their 
    standard deviation relative to the close neighborhood.
    2 - To add more weigt to the peaks because it is often not well fitted
    
    """
        
    # If the non-nan x_coors are inferior to the nb_neighb
    if len(x_coors)<nb_neighb:
        # This can happend on the edges of the frame
        return np.ones(x_coors.shape)
    
    nb_neighb = nb_neighb//2
    weigths = []
    idx_bad_neib = []
    highest_peaks=np.percentile(x_coors, 99.9)
    
    # For each value of x_coors (we do not considere edges)
    for ii in range(0, len(x_coors)):
        
        # Get the neighborhood
        start = np.max((ii-nb_neighb, 0))
        end   = np.min((ii+nb_neighb, len(x_coors)-1))
        neighborhood_x = x_coors[start : end]
        
        # Compute mean & std
        std_neighb = np.std(neighborhood_x)
        mean_neighb = np.mean(neighborhood_x)

        # Check if the candidate match the threshold
        is_good_neighbor = (x_coors[ii] < mean_neighb + std_ratio*std_neighb) *\
                            (x_coors[ii] > mean_neighb - std_ratio*std_neighb)
                    
        # Weight the candidate accordingly
        if x_coors[ii] > highest_peaks :
            weigths.append(2)
        elif is_good_neighbor == True: 
            weigths.append(1.)
        else:
            weigths.append( 0. )
            idx_bad_neib.append(ii)
            
    if show:
        plt.figure("Identifying outliers")
        plt.plot(x_coors)
        for idx in idx_bad_neib:
            plt.plot([idx], [x_coors[idx]], "ro")
            plt.annotate(np.round(weigths[idx], 3), (idx, x_coors[idx]))

    return np.array(weigths)

# %%

def create_spline_cube(cube_frame, centers_lists, center_sci, s=10, k=5, show_slicey=None, frame_sci=None):
    """"Takes all dither put in argument to build the spline of the log10 of the point. The selection of the dither must have been done ahead
    Look at select_dither_type / select_dither_by_postion before 
    
    K represent the degree of the polynomial
    S represent the smoothing factor. The lower is S, the more details are kept.
    
    
    
    """
    from scipy.interpolate import UnivariateSpline

    if show_slicey is True : show_slicey=512 # Default slice if not specified

    # X point to interpolate at the position of the sci data
    nb_dither, y_len, x_len = cube_frame.shape
    ((x_sci_l,y_sci_l), (x_sci_r,y_sci_r)) = center_sci
    
    t_sci = np.linspace(0, x_len-1, x_len)
    cube_spline = np.zeros((y_len,x_len))
    list_spline_funct = []

    # For All row y, create the slice and store it in cube_spline
    for slicey in range(y_len):
        
        all_slice_row_i = []
        x_slice_row_i = []

        #__________________________________________________
        ## Step 1, get all of the dither point of the row with their xshift
        for kk in range(len(cube_frame)):
                    
            frame_i = cube_frame[kk]
    
            (shiftx_l,shifty_l), (shiftx_r,shifty_r) = centers_lists[kk]
    
            slice_i = frame_i[slicey,:]
                            
            x_axis_shifted_left = np.linspace(0, len(slice_i)//2-1, len(slice_i)//2) - shiftx_l + x_sci_l
            x_axis_shifted_right = np.linspace(0, len(slice_i)//2-1, len(slice_i)//2) - shiftx_r + x_len//2 + x_sci_r
            x_axis_shifted = np.concatenate( (x_axis_shifted_left,x_axis_shifted_right) )
            
            # Concatene the dither opint for interpolation 
            all_slice_row_i = np.concatenate((all_slice_row_i, slice_i))
            x_slice_row_i = np.concatenate((x_slice_row_i, x_axis_shifted))

        #__________________________________________________
        ## Step 2, interpolate
        
        # Get the log10 so polynomial fit better
        logslice = np.log10(1+all_slice_row_i)
        #logslice = all_slice_type_i
       
        # The funtion required the x_axis to be sorted        
        indices = np.argsort(x_slice_row_i)
        x_slice_row_i_sorted = x_slice_row_i[indices]
        logslice_sorted= logslice[indices]
        
        # And also to not have repeated x element.
        _ , indices  = np.unique(x_slice_row_i_sorted, return_inverse=True)
        x_slice_row_i_sorted_unique = x_slice_row_i_sorted[indices]
        logslice_sorted_unique = logslice_sorted[indices]
    
        # And no Nans nor Infs
        indices = np.isfinite(logslice_sorted_unique)
        logslice_sorted_unique = logslice_sorted_unique[indices]
        x_slice_row_i_sorted_unique = x_slice_row_i_sorted_unique[indices]
         
            
        # Set weight to avoid outliers problems:
        if show_slicey == slicey :  
            weights = set_weights(logslice_sorted_unique, show=True)
        else : weights = set_weights(logslice_sorted_unique)

        if len(logslice_sorted_unique)<2:
            print("Warning, a slice containe only Nans or negative values")
            print("Model set to f(x)=Nan.")
            splin = lambda x: np.nan
        else:

            # Compute the fuction spline(x) = y, to model the row curve
            splin = UnivariateSpline(x_slice_row_i_sorted_unique,
                                      logslice_sorted_unique, 
                                      w = weights,
                                      s=s, k=k, ext='const')
        
        #__________________________________________________
        ## Step 3, store the function and repeate for next row
        cube_spline[slicey]= splin(t_sci)
        list_spline_funct.append(splin)
        
        if show_slicey == slicey :             

            import matplotlib.pyplot as plt
            
            plt.figure("Spline interpolation -- Slice "+str(slicey) +"and model error")
            
            model_at_ref  = splin(x_slice_row_i_sorted)
            residual = logslice_sorted - model_at_ref
            slice_sci = frame_sci[slicey,:]
            slice_sci = np.log10(1+slice_sci)
            
            plt.plot(t_sci, splin(t_sci), "-",color="tab:orange", alpha=0.5, label="Model, s="+str(s)+", k="+str(k))
            plt.plot(x_slice_row_i_sorted, logslice_sorted, "+", alpha=0.5,color="tab:blue", label="Point cloud")
            plt.plot(x_slice_row_i_sorted, residual, color="tab:red", alpha=0.5, label="Model Error")
            plt.legend()

            # plt.figure("Spline interpolation -- Model  "+str(slicey))

            # #plt.plot(t_sci, splin(t_sci), "-",color="tab:orange", alpha=0.5, label="Model")
            # plt.plot(x_slice_row_i_sorted, logslice_sorted, "+",color="tab:blue", alpha=0.1, label="Model")
            # plt.plot(splin(t_sci), ".-",color="tab:orange", alpha=0.5, label="Model")
            # plt.plot(slice_sci, ".-", alpha=0.5,color="tab:pink", label="Point cloud sci")

            # plt.legend()
            
    return cube_spline, list_spline_funct

# %% 

def re_scale_spline_frame(cube_spline, frame_sci, fitting_window_left=None, fitting_window_right=None, x0=[1, 0], show=False):    
    
    y_len, x_len = frame_sci.shape
    model_frame = np.zeros(frame_sci.shape)
    
    mid = x_len//2
    
    for ii in range(y_len):
        splice_slice = cube_spline[ii, :]
        slice_i = np.log10(1+frame_sci[ii,:])
    
        splice_slice_rescaled_left = re_scale_slice(slice_i[:mid], splice_slice[:mid], fitting_window_left, x0)
        splice_slice_rescaled_right = re_scale_slice(slice_i[mid:], splice_slice[mid:], fitting_window_right, x0)

        model_frame[ii, :] = np.concatenate( (splice_slice_rescaled_left, splice_slice_rescaled_right) )


    if show : 

        residual = np.log10(1+frame_sci) - model_frame
                
        id_slice_y=512

        fig, axd = plt.subplots(1,1, num="Spline model sutraction")
        
        slice_i = np.log10(1+frame_sci[id_slice_y, :])
        splice_slice = model_frame[id_slice_y, :]
        axd.plot(splice_slice, color="lightgrey", alpha=0.5, label="Spline")
        axd.plot(slice_i, "+", color="tab:blue", alpha=0.5, label="Science Point Cloud")
        axd.plot(residual[id_slice_y, :], color="tab:red", alpha=0.5, label="Residual")
        plt.legend()

    return model_frame


def re_scale_slice(slice_dest, slice_to_re_scale, fitting_window=None, x0=[1, 0]):
   
    from scipy.optimize import minimize
    if fitting_window==None:fitting_window = range(0,len(slice_dest))
    
    # Get the window 
    points_model = slice_to_re_scale[fitting_window]
    points_science = slice_dest[fitting_window]
   
    # Remove the Nans : 
    points_model = points_model[~np.isnan(points_science)]
    points_science = points_science[~np.isnan(points_science)]

    # The loss function
    opti_f = lambda param : np.sum( abs(param[0] * points_model + param[1] - points_science) )

    # The minimizer
    res = minimize(opti_f, x0, method='Nelder-Mead', tol=1e-6)
   
    # Get the result
    scale_factor_val,  base = res['x']
    
    # Directly return the re_scaled vector
    return scale_factor_val * slice_to_re_scale + base

# %% 

def re_scale_shift_spline_frame(cube_spline, frame_sci, fitting_window_left=None, fitting_window_right=None, x0=[1, 0, 0], show=False):    
    
    y_len, x_len = frame_sci.shape
    model_frame = np.zeros(frame_sci.shape)
    
    mid = x_len//2
    
    for ii in range(y_len):
        spline_fnct = cube_spline[ii]
        slice_i = np.log10(1+frame_sci[ii,:])
    
        splice_slice_rescaled_left = re_scale_and_shift_slice(slice_i[:mid], spline_fnct, fitting_window_left, x0, side="l")
        splice_slice_rescaled_right = re_scale_and_shift_slice(slice_i[mid:], spline_fnct, fitting_window_right, x0, side="r")

        model_frame[ii, :] = np.concatenate( (splice_slice_rescaled_left, splice_slice_rescaled_right) )


    if show : 

        residual = np.log10(1+frame_sci) - model_frame
                
        id_slice_y=512

        fig, axd = plt.subplots(1,1, num="Spline model sutraction")
        
        slice_i = np.log10(1+frame_sci[id_slice_y, :])
        splice_slice = model_frame[id_slice_y, :]
        splice_slice_no_rescale = cube_spline[id_slice_y](np.arange(0, len(slice_i)))

        
        axd.plot(splice_slice, color="tab:blue", alpha=0.2, label="Spline re-scaled")
        axd.plot(splice_slice_no_rescale, color="lightgrey", alpha=0.5, label="Spline")

        axd.plot(slice_i, "+", color="tab:blue", alpha=0.5, label="Science Point Cloud")
        axd.plot(residual[id_slice_y, :], color="tab:red", alpha=0.5, label="Residual")
        plt.legend()

    return model_frame

def re_scale_and_shift_slice(slice_dest, spline, fitting_window=None, x0=[1, 0, 0], side="l"):
   
    from scipy.optimize import minimize
    if fitting_window==None:fitting_window = range(0,len(slice_dest))
    
   
    # x axis to compute the all spline 
    if side=="l": 
        t_sci= np.arange(0, len(slice_dest))
    else: 
        t_sci= np.arange(len(slice_dest), 2*len(slice_dest))

    # Get the window 
    points_science = slice_dest[fitting_window]
    t_sci_fitting_window = t_sci[fitting_window]

    # Remove the Nans : 
    fitting_window_no_nan = t_sci_fitting_window[np.where(~np.isnan(points_science))]
    points_science = points_science[~np.isnan(points_science)]
    
    # If too much Nan's (wich can happen on edges, do not rescale.)
    if len(points_science)<=2: 
        return spline(t_sci)
    
    # The loss function
    opti_f = lambda param : np.sum( abs(param[0] * spline(fitting_window_no_nan + param[2]) + param[1] - points_science) )

    # The minimizer
    res = minimize(opti_f, x0, method='Nelder-Mead', tol=1e-6)
   
    # Get the result
    scale_factor_val,  base, offset = res['x']
    
    return scale_factor_val * spline(t_sci + offset) + base


# %% Metrics and vizualization function

def show_data_point(cube_frame, centers_list, numdi_list, slicey=512, xlim=None, chose_1color=None, form="+", new_figure=True):
    
    """ Plot a data point """

    nb_dither, y_len, x_len = cube_frame.shape
    mid = x_len//2
    
    if new_figure: plt.figure("Data point plot")
    
    for kk in range(nb_dither):
                
        frame_i = cube_frame[kk]
        label = numdi_list[kk] if isinstance(numdi_list, list or tuple) else numdi_list
        color = chose_1color if chose_1color else colors[kk]
       
        (shiftx_l,shifty_l), (shiftx_r,shifty_r) = centers_list[kk]

        slice_i = frame_i[slicey,:]
                        
        x_axis_shifted_left = np.linspace(0, mid-1, mid) - shiftx_l
        x_axis_shifted_right = np.linspace(mid, 2*mid, mid) - shiftx_r
        x_axis_shifted = np.concatenate( (x_axis_shifted_left,x_axis_shifted_right) )
        
        if isinstance(numdi_list, list or tuple) or kk==0:
            plt.plot(x_axis_shifted, slice_i, form, color=color, label=label, alpha=0.8)

        else:
            # To avoid repeting labels if they all have the same 
            plt.plot(x_axis_shifted, slice_i, form, color=color, alpha=0.8)
        plt.plot(x_axis_shifted, slice_i, "-", color=color, alpha=0.1)
        
    plt.legend()
    plt.xlim(xlim)

    plt.yscale('log')
        
def plot_dither_slice_by_postion(cube_frame, centers_lists, show=False):
    """"Select the list of dither that matches best the science data"""
     
    plt.figure("Slices for dither from the same row")
     
    # For each frame in the provided cube, check if compute beta distance
    beta_dist = []
    for kk in range(len(cube_frame)):
       
        (shiftx_l,shifty_l), (shiftx_r,shifty_r) = centers_lists[kk]

        beta_dist.append(abs(shifty_l))
        
    ## -- Plot the 2 row. Sort them bbeta dist and
    idc = np.argsort(beta_dist)
    
    show_data_point( cube_frame[idc[0:3], :, :], centers_lists[idc[0:3]], "row 1",  chose_1color=colors[0], new_figure=True)
    show_data_point( cube_frame[idc[3:6], :, :], centers_lists[idc[3:6]], "row 2",  chose_1color=colors[1], new_figure=False)
    show_data_point( cube_frame[idc[6:], :, :], centers_lists[idc[6:]], "row 3",  chose_1color=colors[2], new_figure=False)


# %% Can be use to have an interactive plot with a slider. Copy past in main .py file to use
# Not as function because it would required a little trick to not loose handle with the slider,
# I did not made it yet (and I will probably never do)
"""
    from matplotlib.widgets import Slider
    import matplotlib.pyplot as plt
    
    id_slice_y=512

    axd = plt.subplots_adjust(bottom=0.35)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03])
    amp_slider = Slider(
        ax=axamp,
        label="Beta cut",
        valmin=0,
        valmax=1024,
        valinit=512,
        orientation="horizontal"
    )
    
    axd = plt.subplot(1,1,1)

    slice_i = np.log10(1+frame_sci[id_slice_y, :])
    splice_slice = model_frame[id_slice_y, :]
    axd.plot(splice_slice, color="lightgrey", alpha=0.5, label="Spline")
    axd.plot(slice_i, "+", color="tab:blue", alpha=0.5, label="Science Point Cloud")
    axd.plot(residual[id_slice_y, :], color="tab:red", alpha=0.5, label="Residual")

    plt.legend()
    
    def update(val):
        axd.cla()
        id_slice_y = int(val)
        slice_i = np.log10(1+frame_sci[id_slice_y, :])
        splice_slice = model_frame[id_slice_y, :]
        axd.plot(splice_slice, color="lightgrey", alpha=0.5, label="Spline")
        axd.plot(slice_i, "+", color="tab:blue", alpha=0.5, label="Science Point Cloud")
        axd.plot(residual[id_slice_y, :], color="tab:red", alpha=0.5, label="Residual")
        plt.legend()

    amp_slider.on_changed(update)
"""
