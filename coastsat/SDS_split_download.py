import os
import ee
import math
import time
import pytz
import pickle
import rasterio

import numpy as np

from datetime import datetime
from rasterio.merge import merge

from coastsat import SDS_tools
from coastsat.SDS_download import (authenticate_and_initialize, check_images_available, get_s2cloudless, adjust_polygon,
                                   download_tif, warp_image_to_target, get_metadata)


METERS_PER_DEGREE = 111320


def split_area(polygon, sat='S2'):
    aux_left_bound, aux_top_bound = polygon[0]
    aux_right_bound, aux_bot_bound = polygon[2]

    left_bound = min(aux_left_bound, aux_right_bound)
    right_bound = max(aux_left_bound, aux_right_bound)
    bot_bound = min(aux_top_bound, aux_bot_bound)
    top_bound = max(aux_top_bound, aux_bot_bound)

    h_dist = abs(left_bound - right_bound) * METERS_PER_DEGREE
    v_dist = abs(top_bound - bot_bound) * METERS_PER_DEGREE

    if sat == 'S2' or 'COPERNICUS' in sat:
        h_parts = math.ceil(h_dist / 8000)
        v_parts = math.ceil(v_dist / 8000)
    else:
        h_parts = math.ceil(h_dist / 24000)
        v_parts = math.ceil(v_dist / 24000)

    h_offset = h_dist / h_parts if h_parts > 0 else h_dist
    v_offset = v_dist / v_parts if v_parts > 0 else v_dist

    poly_matrix = [[0] * h_parts for _ in range(v_parts)]
    for h in range(h_parts):
        for v in range(v_parts):
            lb = left_bound + h_offset * h / METERS_PER_DEGREE
            rb = left_bound + h_offset * (h + 1) / METERS_PER_DEGREE
            bb = bot_bound + v_offset * v / METERS_PER_DEGREE
            tb = bot_bound + v_offset * (v + 1) / METERS_PER_DEGREE
            poly = [
                [lb, tb],
                [rb, tb],
                [rb, bb],
                [lb, bb],
                [lb, tb]
            ]
            poly_matrix[v_parts - (v + 1)][h] = poly

    return np.array(poly_matrix) if h_parts > 0 or v_parts > 0 else np.array([polygon])


def merge_rasters(raster_list, output_path, method='first'):
    """
    Merge a list of raster (.tif) files into a single raster.

    Parameters:
    - raster_list: list of file paths to input rasters.
    - output_path: file path for the merged output raster.
    - method: how to solve overlaps ('first', 'last', 'min', 'max', 'mean').
    """
    if len(raster_list) == 1:
        os.rename(raster_list[0], output_path)
        return 0

    src_files_to_mosaic = []
    for fp in raster_list:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Merge rasters
    mosaic, out_transform = merge(src_files_to_mosaic, method=method)

    # Copy metadata of first raster and update
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # Write merged raster to disk
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close opened datasets
    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged raster created at {output_path}")


def build_list_structure(n, m, value=None):
    if value is not None and value == 'list':
        value = []
    list_structure = [[value] * m for _ in range(n)]
    return list_structure


def download_image_ee(i_down, j_down, id_image, down_struct, sat_name, poly, site_name):
    # QA band for each satellite mission
    qa_band_Landsat = 'QA_PIXEL'
    qa_band_S2 = 'QA60'
    # the cloud mask band for Sentinel-2 images is the s2cloudless probability
    bands_dict = {'L5': ['B1', 'B2', 'B3', 'B4', 'B5', qa_band_Landsat],
                  'L7': ['B1', 'B2', 'B3', 'B4', 'B5', qa_band_Landsat],
                  'L8': ['B2', 'B3', 'B4', 'B5', 'B6', qa_band_Landsat],
                  'L9': ['B2', 'B3', 'B4', 'B5', 'B6', qa_band_Landsat],
                  'S2': ['B2', 'B3', 'B4', 'B8', 's2cloudless', 'B11', qa_band_S2]}
    # select bands for satellite sensor
    bands_id = bands_dict[sat_name]

    all_names = []  # list for detecting duplicates
    suffix = '.tif'

    image_ee = down_struct['image_ee'][i_down][j_down][id_image]
    im_date = down_struct['im_date'][id_image]
    tilename = down_struct['tilename'][id_image]
    im_meta_id = down_struct['im_meta_id'][id_image]
    filepaths = down_struct['filepaths']

    # download the images as .tif files
    bands = dict([])
    im_fn = dict([])
    # first delete dimensions key from dictionary
    # otherwise the entire image is extracted (don't know why)
    im_bands = image_ee.getInfo()['bands']
    for j in range(len(im_bands)):
        if 'dimensions' in im_bands[j].keys():
            del im_bands[j]['dimensions']

    fp_ms = filepaths[1]
    fp_mask = None
    temp_fp_ms = os.path.join(fp_ms, 'temp_downloads')
    file_name_extra = None
    temp_ms, temp_mask, temp_extra = None, None, None

    # =============================================================================================#
    # Landsat 5 download
    # =============================================================================================#
    if sat_name == 'L5':
        fp_mask = filepaths[2]
        # select multispectral bands
        bands['ms'] = [im_bands[_] for _ in range(len(im_bands)) if im_bands[_]['id'] in bands_id]
        # adjust polygon to match image coordinates so that there is no resampling
        proj = image_ee.select('B1').projection()
        ee_region = adjust_polygon(poly, proj)
        # download .tif from EE (one file with ms bands and one file with QA band)
        count = 0
        while True:
            try:
                fn_ms, fn_QA = download_tif(image_ee, ee_region, bands['ms'], fp_ms, sat_name)
                break
            except Exception as e:
                print('\nDownload failed, trying again...')
                print(f"The error that occurred: {e}")
                time.sleep(60)
                count += 1
                if count > 100:
                    raise Exception('Too many attempts, crashed while downloading image %s' % im_meta_id)
                else:
                    continue

        # create filename for image
        for key in bands.keys():
            im_fn[key] = im_date + '_' + sat_name + '_' + tilename + '_' + site_name + '_' + key + suffix
        # if multiple images taken at the same date add 'dupX' to the name (duplicate number X)
        duplicate_counter = 0
        while im_fn['ms'] in all_names:
            duplicate_counter += 1
            for key in bands.keys():
                im_fn[key] = im_date + '_' + sat_name + '_' + tilename + '_' \
                             + site_name + '_' + key \
                             + '_dup%d' % duplicate_counter + suffix
        im_fn['mask'] = im_fn['ms'].replace('_ms', '_mask')
        filename_ms = im_fn['ms']
        all_names.append(im_fn['ms'])

        # resample ms bands to 15m with bilinear interpolation
        fn_in = fn_ms
        fn_target = fn_ms
        fn_out = os.path.join(fp_ms, im_fn['ms'])
        temp_ms = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        warp_image_to_target(fn_in, temp_ms, fn_target, double_res=True, resampling_method='bilinear')

        # resample QA band to 15m with nearest-neighbour interpolation
        fn_in = fn_QA
        fn_target = fn_QA
        fn_out = os.path.join(fp_mask, im_fn['mask'])
        temp_mask = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        warp_image_to_target(fn_in, temp_mask, fn_target, double_res=True, resampling_method='near')

        # delete original downloads
        for _ in [fn_ms, fn_QA]: os.remove(_)

    # =============================================================================================#
    # Landsat 7, 8 and 9 download
    # =============================================================================================#
    elif sat_name in ['L7', 'L8', 'L9']:
        fp_pan = filepaths[2]
        fp_mask = filepaths[3]

        # select bands (multispectral and panchromatic)
        bands['ms'] = [im_bands[_] for _ in range(len(im_bands)) if im_bands[_]['id'] in bands_id]
        bands['pan'] = [im_bands[_] for _ in range(len(im_bands)) if im_bands[_]['id'] in ['B8']]
        # adjust polygon for both ms and pan bands
        proj_ms = image_ee.select('B1').projection()
        proj_pan = image_ee.select('B8').projection()
        ee_region_ms = adjust_polygon(poly, proj_ms)
        ee_region_pan = adjust_polygon(poly, proj_pan)

        # download both ms and pan bands from EE
        count = 0
        while True:
            try:
                fn_ms, fn_QA = download_tif(image_ee, ee_region_ms, bands['ms'], fp_ms, sat_name)
                fn_pan = download_tif(image_ee, ee_region_pan, bands['pan'], fp_pan, sat_name)
                break
            except Exception as e:
                print('\nDownload failed, trying again...')
                print(f"The error that occurred: {e}")
                time.sleep(60)
                count += 1
                if count > 100:
                    raise Exception('Too many attempts, crashed while downloading image %s' % im_meta_id)
                else:
                    continue

        # create filename for both images (ms and pan)
        for key in bands.keys():
            im_fn[key] = im_date + '_' + sat_name + '_' + tilename + '_' + site_name + '_' + key + suffix
        # if multiple images taken at the same date add 'dupX' to the name (duplicate number X)
        duplicate_counter = 0
        while im_fn['ms'] in all_names:
            duplicate_counter += 1
            for key in bands.keys():
                im_fn[key] = im_date + '_' + sat_name + '_' + tilename + '_' \
                             + site_name + '_' + key \
                             + '_dup%d' % duplicate_counter + suffix
        im_fn['mask'] = im_fn['ms'].replace('_ms', '_mask')
        filename_ms = im_fn['ms']
        all_names.append(im_fn['ms'])

        file_name_extra = os.path.join(fp_pan, im_fn['pan'])

        # resample the ms bands to the pan band with bilinear interpolation (for pan-sharpening later)
        fn_in = fn_ms
        fn_target = fn_pan
        fn_out = os.path.join(fp_ms, im_fn['ms'])
        temp_ms = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        warp_image_to_target(fn_in, temp_ms, fn_target, double_res=False, resampling_method='bilinear')

        # resample QA band to the pan band with nearest-neighbour interpolation
        fn_in = fn_QA
        fn_target = fn_pan
        fn_out = os.path.join(fp_mask, im_fn['mask'])
        temp_mask = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        warp_image_to_target(fn_in, temp_mask, fn_target, double_res=False, resampling_method='near')

        # rename pan band
        fn_out = im_fn['pan']
        temp_extra = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        temp_extra = os.path.join(fp_pan, temp_extra)
        try:
            os.rename(fn_pan, temp_extra)
        except:
            os.remove(temp_extra)
            os.rename(fn_pan, temp_extra)
            # delete original downloads
        for _ in [fn_ms, fn_QA]: os.remove(_)

    # =============================================================================================#
    # Sentinel-2 download
    # =============================================================================================#
    elif sat_name in ['S2']:
        fp_swir = filepaths[2]
        fp_mask = filepaths[3]

        # select bands (10m ms RGB+NIR+s2cloudless, 20m SWIR1, 60m QA band)
        bands['ms'] = [im_bands[_] for _ in range(len(im_bands)) if im_bands[_]['id'] in bands_id[:5]]
        bands['swir'] = [im_bands[_] for _ in range(len(im_bands)) if im_bands[_]['id'] in bands_id[5:6]]
        bands['mask'] = [im_bands[_] for _ in range(len(im_bands)) if im_bands[_]['id'] in bands_id[-1:]]
        # adjust polygon for both ms and pan bands
        proj_ms = image_ee.select('B1').projection()
        proj_swir = image_ee.select('B11').projection()
        proj_mask = image_ee.select('QA60').projection()
        ee_region_ms = adjust_polygon(poly, proj_ms)
        ee_region_swir = adjust_polygon(poly, proj_swir)
        ee_region_mask = adjust_polygon(poly, proj_mask)
        # download the ms, swir and QA bands from EE
        count = 0
        while True:
            try:
                fn_ms = download_tif(image_ee, ee_region_ms, bands['ms'], fp_ms, sat_name)
                fn_swir = download_tif(image_ee, ee_region_swir, bands['swir'], fp_swir, sat_name)
                fn_QA = download_tif(image_ee, ee_region_mask, bands['mask'], fp_mask, sat_name)
                break
            except Exception as e:
                print('\nDownload failed, trying again...')
                print(f"The error that occurred: {e}")
                time.sleep(60)
                count += 1
                if count > 100:
                    raise Exception('Too many attempts, crashed while downloading image %s' % im_meta_id)
                else:
                    continue

                    # create filename for the three images (ms, swir and mask)
        for key in bands.keys():
            im_fn[key] = im_date + '_' + sat_name + '_' + tilename + '_' + site_name + '_' + key + suffix
        # if multiple images taken at the same date add 'dupX' to the name (duplicate)
        duplicate_counter = 0
        while im_fn['ms'] in all_names:
            duplicate_counter += 1
            for key in bands.keys():
                im_fn[key] = im_date + '_' + sat_name + '_' + tilename + '_' \
                             + site_name + '_' + key \
                             + '_dup%d' % duplicate_counter + suffix
        filename_ms = im_fn['ms']
        all_names.append(im_fn['ms'])

        file_name_extra = os.path.join(fp_swir, im_fn['swir'])

        # resample the 20m swir band to the 10m ms band with bilinear interpolation
        fn_in = fn_swir
        fn_target = fn_ms
        fn_out = os.path.join(fp_swir, im_fn['swir'])
        temp_extra = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        warp_image_to_target(fn_in, temp_extra, fn_target, double_res=False, resampling_method='bilinear')

        # resample 60m QA band to the 10m ms band with nearest-neighbour interpolation
        fn_in = fn_QA
        fn_target = fn_ms
        fn_out = os.path.join(fp_mask, im_fn['mask'])
        temp_mask = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        warp_image_to_target(fn_in, temp_mask, fn_target, double_res=False, resampling_method='near')

        # delete original downloads
        for _ in [fn_swir, fn_QA]: os.remove(_)
        # rename the multispectral band file
        fn_out = im_fn['ms']
        temp_ms = fn_out.replace('.tif', f'_temp_{i_down}_{j_down}.tif')
        temp_ms = os.path.join(fp_ms, temp_ms)
        os.rename(fn_ms, os.path.join(fp_ms, temp_ms))

    return fp_ms, temp_ms, temp_mask, temp_extra, os.path.join(fp_ms, im_fn['ms']), os.path.join(fp_mask, im_fn['mask']), file_name_extra


def retrieve_images(inputs):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8, Landsat 9 and Sentinel-2
    covering the area of interest and acquired between the specified dates.
    The downloaded images are in .TIF format and organised in subfolders, divided
    by satellite mission. The bands are also subdivided by pixel resolution.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # initialise connection with GEE server
    authenticate_and_initialize()

    # splitting original polygon into smaller parts
    poly_parts = split_area(inputs['polygon'][0], sat='S2' if 'S2' in inputs['sat_list'] else 'L')
    poly_n, poly_m = poly_parts.shape[:2]

    # generate metadata structure
    download_metadata = {k: {} for k in inputs['sat_list']}
    for k, v in download_metadata.items():
        download_metadata[k]['image_ee'] = build_list_structure(poly_n, poly_m, value='list')
        download_metadata[k]['im_date'] = []
        download_metadata[k]['tilename'] = []
        download_metadata[k]['im_meta_id'] = []
        download_metadata[k]['filepaths'] = None
    download_metadata['polygons'] = poly_parts
    download_metadata['sitename'] = inputs['sitename']

    inputs_bu = inputs.copy()
    for i_poly_p in range(poly_n):
        for j_poly_p in range(poly_m):
            poly_p = poly_parts[i_poly_p][j_poly_p]
            inputs = inputs_bu.copy()
            inputs['polygon'] = poly_p.tolist()

            # check image availabiliy and retrieve list of images
            im_dict_T1, im_dict_T2 = check_images_available(inputs)

            # if user also wants to download T2 images, merge both lists
            if 'include_T2' in inputs.keys():
                for key in inputs['sat_list']:
                    if key in ['S2', 'L9']:
                        continue
                    else:
                        im_dict_T1[key] += im_dict_T2[key]

            # for S2 get s2cloudless collection for advanced cloud masking
            if 'S2' in inputs['sat_list'] and len(im_dict_T1['S2']) > 0:
                im_dict_s2cloudless = get_s2cloudless(im_dict_T1['S2'], inputs)

            # create a new directory for this site with the name of the site
            im_folder = os.path.join(inputs['filepath'], inputs['sitename'])
            if not os.path.exists(im_folder): os.makedirs(im_folder)

            # main loop to download the images for each satellite mission
            print('\nDownloading images:')
            for satname in im_dict_T1.keys():

                # print how many images will be downloaded for the users
                print('%s: %d images' % (satname, len(im_dict_T1[satname])))

                # create subfolder structure to store the different bands
                filepaths = SDS_tools.create_folder_structure(im_folder, satname)

                # loop through each image
                for i in range(len(im_dict_T1[satname])):

                    # get image metadata
                    im_meta = im_dict_T1[satname][i]

                    # get time of acquisition (UNIX time) and convert to datetime
                    t = im_meta['properties']['system:time_start']
                    im_timestamp = datetime.fromtimestamp(t / 1000, tz=pytz.utc)
                    im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')

                    # special case for L7 as it had the Scan Line Correction failure
                    if satname == 'L7':
                        # skip L7 after 2022 as L9 has replaced it
                        if im_timestamp.year >= 2022:
                            continue
                        # optionally, skip L7 after Scan-Line-Correction failure
                        if 'skip_L7_SLC' in inputs.keys():
                            if inputs['skip_L7_SLC']:
                                if im_timestamp >= pytz.utc.localize(datetime(2003, 5, 31)):
                                    continue

                    # get epsg code
                    im_epsg = int(im_meta['bands'][0]['crs'][5:])

                    # get geometric accuracy, radiometric quality and tilename for Landsat
                    if satname in ['L5', 'L7', 'L8', 'L9']:
                        if 'GEOMETRIC_RMSE_MODEL' in im_meta['properties'].keys():
                            acc_georef = im_meta['properties']['GEOMETRIC_RMSE_MODEL']
                        else:
                            acc_georef = 12  # average georefencing error across Landsat collection (RMSE = 12m)
                        # add radiometric quality [image_quality 1-9 for Landsat]
                        if satname in ['L5', 'L7']:
                            rad_quality = im_meta['properties']['IMAGE_QUALITY']
                        elif satname in ['L8', 'L9']:
                            rad_quality = im_meta['properties']['IMAGE_QUALITY_OLI']
                        # add tilename (path/row)
                        tilename = '%03d%03d' % (im_meta['properties']['WRS_PATH'], im_meta['properties']['WRS_ROW'])

                    # get geometric accuracy, radiometric quality and tilename for S2
                    elif satname in ['S2']:
                        # Sentinel-2 products don't provide a georeferencing accuracy (RMSE as in Landsat)
                        # but they have a flag indicating if the geometric quality control was PASSED or FAILED
                        # if passed a value of 1 is stored if failed a value of -1 is stored in the metadata
                        # check which flag name is used for the image as it changes for some reason in the archive
                        flag_names = ['GEOMETRIC_QUALITY_FLAG', 'GEOMETRIC_QUALITY', 'quality_check', 'GENERAL_QUALITY_FLAG']
                        key = []
                        for key in flag_names:
                            if key in im_meta['properties'].keys():
                                break  # use the first flag that is found
                        if len(key) > 0:
                            acc_georef = im_meta['properties'][key]
                        else:
                            print('WARNING: could not find Sentinel-2 geometric quality flag,' +
                                  ' raise an issue at https://github.com/kvos/CoastSat/issues' +
                                  ' and add you inputs in text (not a screenshot pls).')
                            acc_georef = 'PASSED'
                        # add the radiometric image quality ['PASSED' or 'FAILED']
                        flag_names = ['RADIOMETRIC_QUALITY', 'RADIOMETRIC_QUALITY_FLAG']
                        key = []
                        for key in flag_names:
                            if key in im_meta['properties'].keys():
                                break  # use the first flag that is found
                        if len(key) > 0:
                            rad_quality = im_meta['properties'][key]
                        else:
                            print('WARNING: could not find Sentinel-2 geometric quality flag,' +
                                  ' raise an issue at https://github.com/kvos/CoastSat/issues' +
                                  ' and add your inputs in text (not a screenshot pls).')
                            rad_quality = 'PASSED'
                        # add tilename (MGRS name)
                        tilename = im_meta['properties']['MGRS_TILE']

                    # select image by id
                    image_ee = ee.Image(im_meta['id'])

                    # for S2 add s2cloudless probability band
                    if satname == 'S2':
                        if len(im_dict_s2cloudless[i]) == 0:
                            print('Warning: S2cloudless mask for image %s is not available yet, try again tomorrow.' % im_date)
                            continue
                        im_cloud = ee.Image(im_dict_s2cloudless[i]['id'])
                        cloud_prob = im_cloud.select('probability').rename('s2cloudless')
                        image_ee = image_ee.addBands(cloud_prob)

                    # appending downloading metadata to structure
                    download_metadata[satname]['image_ee'][i_poly_p][j_poly_p].append(image_ee)
                    download_metadata[satname]['tilename'].append(tilename)
                    download_metadata[satname]['im_meta_id'].append(im_meta['id'])
                    download_metadata[satname]['im_date'].append(im_date)
                    download_metadata[satname]['filepaths'] = filepaths

    # download images now
    for satname in inputs['sat_list']:
        for id_image in range(len(download_metadata[satname]['tilename'])):
            files_ms = []
            files_mask = []
            files_extra = []
            final_ms = None
            final_mask = None
            final_extra = None
            fp_ms = None
            for i_struct in range(poly_n):
                for j_struct in range(poly_m):
                    fp_ms, ms, mask, extra, final_ms, final_mask, final_extra = download_image_ee(
                        i_struct,
                        j_struct,
                        id_image,
                        download_metadata[satname],
                        satname,
                        download_metadata['polygons'][i_struct][j_struct].tolist(),
                        download_metadata['sitename']
                    )
                    files_ms.append(ms)
                    files_mask.append(mask)
                    files_extra.append(extra)

            # merge results
            merge_rasters(files_ms, final_ms)
            merge_rasters(files_mask, final_mask)
            if files_extra[0] is not None:
                merge_rasters(files_extra, final_extra)

            # delete temp files
            for f in files_ms + files_mask + files_extra:
                if f is not None and os.path.exists(f):
                    os.remove(f)

            # get image dimensions (width and height)
            final_ms = os.path.basename(final_ms)
            image_path = os.path.join(fp_ms, final_ms)
            width, height = SDS_tools.get_image_dimensions(image_path)
            # write metadata in a text file for easy access
            filename_txt = final_ms.replace('_ms', '').replace('.tif', '')
            metadict = {'filename': final_ms, 'tile': tilename, 'epsg': im_epsg,
                        'acc_georef': acc_georef, 'image_quality': rad_quality,
                        'im_width': width, 'im_height': height}
            with open(os.path.join(download_metadata[satname]['filepaths'][0], filename_txt + '.txt'), 'w') as f:
                for key in metadict.keys():
                    f.write('%s\t%s\n' % (key, metadict[key]))
            # print percentage completion for user
            print('\r%d%%' % int((i + 1) / len(im_dict_T1[satname]) * 100), end='')

            print('')

    # once all images have been downloaded, load metadata from .txt files
    metadata = get_metadata(inputs_bu)

    # save metadata dict
    with open(os.path.join(im_folder, inputs['sitename'] + '_metadata' + '.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print('Satellite images downloaded from GEE and save in %s' % im_folder)
    return metadata
