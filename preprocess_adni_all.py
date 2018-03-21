# Script to preprocess the ADNI1 baseline (screening) data
# Requires a directory with the image data
# and the following ADNI tables
# - ADNIMERGE.csv  : Most information is here
# - VITALS.csv  : Some extra infos about congnitive scores
# - DXSUM_PDXCONV_ADNIALL.csv  : Patient weight
# - MRI3META.csv  : For figuring out if images are 3T
# - MRIMETA.csv  : or 1.5T.
#
# The script proprocesses the images with some optional steps
# and also writes a summary_screening.csv file.
# That file merges the infos from the different tables
# 'nan' fields are fields that were empty in the input tables
# 'unknown' fields are fields that could not be found in the tables (i.e. no corresponding rid/viscode combination
#
# The INCLUDE_MISSING_IMAGES_IN_TABLE can be set to true or false to include all images in the csv table
# or only the ones we have images for.
#
# Author:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
#


import pandas as pd
import os
import glob
import datetime
import time
import csv
import shutil
import utils
from subprocess import Popen
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

INCLUDE_MISSING_IMAGES_IN_TABLE = True
DO_ONLY_TABLE = True

bmicdatasets_root = '/usr/bmicnas01/data-biwi-01/'
bmicdatasets_originals = os.path.join(bmicdatasets_root, 'bmicdatasets-originals/Originals/')
bmicdatasets_adni = os.path.join(bmicdatasets_originals, 'ADNI/')
bmicdatasets_adni_tables = os.path.join(bmicdatasets_adni, 'Tables')
bmicdatasets_adni_images = os.path.join(bmicdatasets_adni, 'adni_all_mri/ADNI/')

bmidatasets_mni = os.path.join(bmicdatasets_originals, 'TemplateData/MNI/mni_icbm152_nlin_asym_09a')
mni_template_t1 = os.path.join(bmidatasets_mni, 'mni_icbm152_t1_tal_nlin_asym_09a.nii')

adni_merge_path = os.path.join(bmicdatasets_adni_tables, 'ADNIMERGE.csv')
vitals_path = os.path.join(bmicdatasets_adni_tables, 'VITALS.csv')
diagnosis_path = os.path.join(bmicdatasets_adni_tables, 'DXSUM_PDXCONV_ADNIALL.csv')
mri_3_0_meta_path = os.path.join(bmicdatasets_adni_tables, 'MRI3META.csv')
mri_1_5_meta_path = os.path.join(bmicdatasets_adni_tables, 'MRIMETA.csv')

N4_executable = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4'
robex_executable = '/scratch_net/bmicdl03/software/robex/robex-build/ROBEX'

def date_string_to_seconds(date_str):

    date, time = date_str.split(' ')
    year, month, day = [int(i.split('.')[0]) for i in date.split('-')]
    hours, minutes, secs = [int(i.split('.')[0]) for i in time.split(':')]

    acq_time = datetime.datetime(year, month, day, hours, minutes, secs)

    start_of_time = datetime.datetime(1970,1,1)

    return (acq_time - start_of_time).total_seconds()

#
# def find_by_conditions(pandas_df, condition_dict):
#
#     for ii, (key, value) in enumerate(condition_dict.items()):
#         if ii == 0:
#             conds = pandas_df[key] == value
#         else:
#             conds = conds & (pandas_df[key] == value)
#
#     return pandas_df.loc[conds]



def find_by_conditions(pandas_df, and_condition_dict=None, or_condition_dict=None):

    if and_condition_dict is not None:
        conds_and = True
        for ii, (key, value_list) in enumerate(and_condition_dict.items()):
            if not isinstance(value_list, list):
                value_list = [value_list]
            for value in list(value_list):
                conds_and = conds_and & (pandas_df[key] == value)
    else:
        conds_and = False

    if or_condition_dict is not None:
        conds_or = False
        for ii, (key, value_list) in enumerate(or_condition_dict.items()):
            if not isinstance(value_list, list):
                value_list = [value_list]
            for value in list(value_list):
                conds_or = conds_or | (pandas_df[key] == value)
    else:
        conds_or = True

    conds = conds_and & conds_or

    # logging.info('conds:')
    # logging.info(sum(conds))

    return pandas_df.loc[conds]

def diagnosis_to_3categories_blformat(diag_str):

    if diag_str in ['EMCI', 'LMCI', 'MCI']:
        return 'MCI'
    elif diag_str in ['CN', 'SMC']:
        return 'CN'
    elif diag_str in ['AD', 'Dementia']:
        return 'AD'
    else:
        raise ValueError('Unknown diagnosis: "%s"'% diag_str)


def diagnosis_to_3categories(diag_str):

    if diag_str in [2,4,8]:
        return 'MCI'
    elif diag_str in [1,7,9]:
        return 'CN'
    elif diag_str in [3,5,6]:
        return 'AD'
    elif diag_str in [0]:
        return 'unknown'
    else:
        raise ValueError('Unknown diagnosis: "%s"'% diag_str)

def convert_weight_to_kg(weight, unit):

    if unit == 2:
        return weight
    elif unit == 1:
        return 0.453592*weight
    else:
        return weight  # probably weight unknown = -1


def map_all_baseline_to_bl(list):
    return_list =[]
    for ll in list:
        if ll in ['bl', 'sc', 'scmri']:
            return_list.append('bl')
        else:
            return_list.append(ll)
    return return_list


def all_same(items):
    return all(x == items[0] for x in items)

def do_preprocessing(adnimerge_table_arg,
                     tmp_index,
                     processed_images_folder,
                     summary_csv_file,
                     do_reorientation=False,
                     do_registration=False,
                     do_bias_correction=False,
                     do_cropping=False,
                     do_skull_stripping=False,
                     write_csv=True):

    if do_reorientation | do_registration | do_bias_correction | do_cropping | do_skull_stripping == False:
        do_postprocessing = False
    else:
        do_postprocessing = True

    vitals_table = pd.read_csv(vitals_path)

    mri_3_0_meta_table = pd.read_csv(mri_3_0_meta_path)
    mri_1_5_meta_table = pd.read_csv(mri_1_5_meta_path)

    diagnosis_table = pd.read_csv(diagnosis_path)


    tmp_file_folder = os.path.join(processed_images_folder, 'tmp')
    if do_postprocessing:
        utils.makefolder(tmp_file_folder)

    with open(summary_csv_file, 'w') as csvfile:


        if write_csv:
            csvwriter = csv.writer(csvfile, delimiter=',')

            csvwriter.writerow(['rid', 'phase', 'image_exists', 'site', 'viscode', 'exam_date', 'field_strength', 'diagnosis', 'diagnosis_3cat',
                                'age', 'gender', 'weight',
                                'education', 'ethnicity', 'race', 'apoe4', 'adas13', 'mmse', 'faq', 'counter' ])


        for ii, row in adnimerge_table_arg.iterrows():

            viscode = row['VISCODE']
            # if viscode not in ['bl']:  #  or 'sc']: There are no 'sc' in adnimerge
            #     continue

            rid = row['RID']
            phase = row['COLPROT']

            if phase in ['ADNI3']:
                continue


            site = row['SITE']

            age_at_bl = row['AGE']   # Note ADNIMERGE age is always the same, even for the follow up scans years later
            gender = row['PTGENDER']
            education = row['PTEDUCAT']
            ethnicity = row['PTETHCAT']
            race = row['PTRACCAT']
            apoe4 = row['APOE4']
            adas13 = row['ADAS13']
            mmse = row['MMSE']
            faq = row['FAQ']
            exam_date_adnimerge = row['EXAMDATE']  # Not necessarily the same as the exam date in the MRIMETA files

            diagnosis_row = find_by_conditions(diagnosis_table, and_condition_dict={'RID': rid, 'VISCODE2': viscode})
            if phase == 'ADNI1':
                diagnosis = diagnosis_row['DXCURREN'].values
            else:
                diagnosis = diagnosis_row['DXCHANGE'].values

            print('---- rid %s -----' % rid)
            print(viscode)
            print(diagnosis)

            if len(diagnosis) == 0:
                diagnosis = 0
                if viscode == 'm03':
                    diagnosis_bl = row['DX_bl']
                    diagnosis_3cat = diagnosis_to_3categories_blformat(diagnosis_bl)
                else:
                    diagnosis_3cat = 'unknown'
            else:
                diagnosis = int(diagnosis[0])
                diagnosis_3cat = diagnosis_to_3categories(diagnosis)


            # field_strength = row['FLDSTRENG']  # This field is incomplete, too many nan values

            vitals_row = find_by_conditions(vitals_table, {'RID': rid, 'VISCODE2': 'bl'})  # here also examdates sometimes don't correspond
            if len(vitals_row) == 0:
                vitals_row = find_by_conditions(vitals_table, {'RID': rid, 'VISCODE2': 'sc'})

            assert len(vitals_row) <= 1, 'in vitals table found %d rows for case with rid=%s, and viscode=bl. Expected one.' \
                                         % (len(vitals_row), rid)

            # Getting some vitals information
            if len(vitals_row) == 1:
                weight = vitals_row['VSWEIGHT'].values[0]
                weight_units = vitals_row['VSWTUNIT'].values[0]
                weight = convert_weight_to_kg(weight, weight_units)
            else:
                weight = 'unknown'


            mri_1_5_meta_row = find_by_conditions(mri_1_5_meta_table, and_condition_dict={'RID': rid, 'VISCODE2': viscode})
            if len(mri_1_5_meta_row) == 0 and viscode == 'bl':
                mri_1_5_meta_row = find_by_conditions(mri_1_5_meta_table,
                                                      and_condition_dict={'RID': rid, 'VISCODE2': 'sc'})


            mri_3_0_meta_row = find_by_conditions(mri_3_0_meta_table, and_condition_dict={'RID': rid, 'VISCODE2': viscode})
            if len(mri_3_0_meta_row) == 0 and viscode == 'bl':
                mri_3_0_meta_row = find_by_conditions(mri_3_0_meta_table,
                                                      and_condition_dict={'RID': rid},
                                                      or_condition_dict={'VISCODE2': ['sc', 'scmri']})


            exam_dates = list(mri_1_5_meta_row['EXAMDATE'].values) + list(mri_3_0_meta_row['EXAMDATE'].values)
            field_strengths = [1.5]*len(mri_1_5_meta_row['EXAMDATE']) + [3.0]*len(mri_3_0_meta_row['EXAMDATE'])
            viscodes = list(mri_1_5_meta_row['VISCODE2'].values) + list(mri_3_0_meta_row['VISCODE2'].values)


            subj_subfolder = '%s_S_%s' % (str(site).zfill(3), str(rid).zfill(4))

            # Remove nans from exam dates and corresponding field strengths
            exam_dates_tmp = []
            field_strengths_tmp = []
            viscodes_tmp = []
            for ed, fs, vc in zip(exam_dates, field_strengths, viscodes):
                if str(ed) != 'nan':
                    exam_dates_tmp.append(ed)
                    field_strengths_tmp.append(fs)
                    viscodes_tmp.append(vc)
            exam_dates = exam_dates_tmp
            field_strengths = field_strengths_tmp
            viscodes = viscodes_tmp

            # If all exam dates are the same keep only one
            if len(exam_dates) > 1 and all_same(exam_dates):

                print('Multiple equal exam dates')
                print(field_strengths)

                exam_dates = [exam_dates[0]]
                field_strengths = [field_strengths[0]]
                viscodes = [viscodes[0]]


            # If all there are duplicate viscodes keep the first and say 1.5T because duplicates are almost always 1.5T
            if len(viscodes) > 1 and all_same(map_all_baseline_to_bl(viscodes)):
                print('Identical viscodes')
                print(field_strengths)
                exam_dates = [exam_dates[0]]
                if phase in ['ADNI1', 'ADNIGO']:
                    field_strengths =  [field_strengths[0]]  # 1.5 is always the first item anyways
                else:
                    print('!! Multiple viscodes. Duplicate that was in ADNI2')
                    print(field_strengths)
                    field_strengths = [field_strengths[0]]


            if not len(exam_dates) > 0:
                continue

            # Philips scanners have do not have the gradwarp preprocessed images. I am assuming MT1__N3m is fine even
            # though B1_Correctino is missing.
            # This webpage: http://adni.loni.usc.edu/methods/mri-analysis/mri-pre-processing/ says all files with a N3m
            # in the end are fine to use. I am assuming that MPR____N3 and MPR__GradWarp__N3 also indicate that the
            # whole preprocessing pipeline was applied.
            preproc_subfolders = ['MPR__GradWarp__B1_Correction__N3', 'MPR____N3', 'MT1__N3m', 'MT1__GradWarp__N3m', 'MPR__GradWarp__N3']

            nii_files = []

            for exam_date, field_strength in zip(exam_dates, field_strengths):

                # figure out age:
                # get baseline examdate from adnimerge
                baseline_row = find_by_conditions(adnimerge_table_arg,
                                                  and_condition_dict={'RID': rid},
                                                  or_condition_dict={'VISCODE': ['sc', 'scmri', 'bl']})

                baseline_exam_dates = baseline_row['EXAMDATE'].values

                if len(baseline_exam_dates) <= 0:
                    current_age = 'unknown'
                else:
                    baseline_exam_date = baseline_exam_dates[0]

                    year_diff = int(exam_date.split('-')[0]) - int(baseline_exam_date.split('-')[0])
                    month_diff = int(exam_date.split('-')[1]) - int(baseline_exam_date.split('-')[1])
                    day_diff = int(exam_date.split('-')[2]) - int(baseline_exam_date.split('-')[2])

                    decimal_year_diff = year_diff + (1.0/12)*month_diff + (1.0/(12*30)*day_diff)

                    assert decimal_year_diff >= -0.75, 'Year diff cannot be (too) negative! Was %f' % decimal_year_diff

                    if decimal_year_diff < 0:
                        decimal_year_diff = 0.0

                    current_age = age_at_bl + decimal_year_diff


                for preproc_subfolder in preproc_subfolders:
                    nii_search_str = os.path.join(subj_subfolder, preproc_subfolder, exam_date + '_*', '*/*.nii')
                    nii_files += glob.glob(os.path.join(bmicdatasets_adni_images, nii_search_str))

                # If some files have gradwarp prefer those files
                contains_GradWarp = any(['GradWarp' in ff for ff in nii_files])
                if contains_GradWarp:
                    nii_files = [ff for ff in nii_files if 'GradWarp' in ff]


                # if some files have MT1 and MPR prefer the MT1
                contains_MT1 = any(['MT1' in ff for ff in nii_files])
                if contains_MT1:
                    nii_files = [ff for ff in nii_files if 'MT1' in ff]


                # if some files have B1 correction prefer those
                contains_B1 = any(['B1_Correction' in ff for ff in nii_files])
                if contains_B1:
                    nii_files = [ff for ff in nii_files if 'B1_Correction' in ff]

                image_exists = True if len(nii_files) > 0 else False


                if image_exists:

                    start_time = time.time()

                    if not DO_ONLY_TABLE:

                        nii_use_file = nii_files[0]
                        logging.info(nii_use_file)

                        gz_postfix = '.gz' if do_postprocessing else ''
                        patient_folder = 'rid_%s' % (str(rid).zfill(4))
                        out_file_name = '%s_%sT_%s_rid%s_%s.nii%s' % (phase.lower(),
                                                                      field_strength,
                                                                      diagnosis_3cat,
                                                                      str(rid).zfill(4),
                                                                      viscode,
                                                                      gz_postfix)

                        out_folder = os.path.join(processed_images_folder, patient_folder)
                        utils.makefolder(out_folder)

                        out_file_path = os.path.join(out_folder, out_file_name)

                        if os.path.exists(out_file_path):
                            logging.info('!!! File already exists. Skipping')
                            continue
                        else:
                            logging.info('--- Doing File: %s' % out_file_path)

                        if not do_postprocessing:
                            logging.info('Not doing any preprocessing...')
                            shutil.copyfile(nii_use_file, out_file_path)
                        else:
                            tmp_file_path = os.path.join(tmp_file_folder, 'tmp_rid%s_%s.nii.gz' % (str(rid).zfill(4), str(tmp_index)))
                            shutil.copyfile(nii_use_file, tmp_file_path)

                            if do_reorientation:
                            # fsl orientation enforcing:
                                logging.info('Reorienting to MNI space...')
                                Popen('fslreorient2std {0} {1}'.format(tmp_file_path, tmp_file_path), shell=True).communicate()

                            if do_cropping:

                                # field of view cropping
                                logging.info('Cropping the field of view...')
                                Popen('robustfov -i {0} -r {1}'.format(tmp_file_path, tmp_file_path), shell=True).communicate()

                            if do_bias_correction:
                                # bias correction with N4:
                                logging.info('Bias correction...')
                                Popen('{0} {1} {2}'.format(N4_executable, tmp_file_path, tmp_file_path),
                                      shell=True).communicate()

                            if do_registration:

                                # registration with flirt to MNI 152:
                                logging.info('Registering the structural image...')
                                Popen(
                                    'flirt -in {0} -ref {1} -out {2} -searchrx -45 45 -searchry -45 45 -searchrz -45 45 -dof 7'.format(
                                        tmp_file_path, mni_template_t1, tmp_file_path), shell=True).communicate()

                            if do_skull_stripping:

                                # skull stripping with bet2
                                logging.info('Skull stripping...')
                                # Popen('bet {0} {1} -R -f 0.5 -g 0'.format(tmp_file_path, tmp_file_path), shell=True).communicate()  # bet was not robust enough
                                Popen('{0} {1} {2} -R -f 0.5 -g 0'.format(robex_executable, tmp_file_path, tmp_file_path), shell=True).communicate()
                                logging.info('Finished.')


                            logging.info('Copying tmp file: %s, to output: %s' % (tmp_file_path, out_file_path))
                            shutil.copyfile(tmp_file_path, out_file_path)


                    if write_csv:
                        csvwriter.writerow([rid, phase, image_exists, site, viscode, exam_date, field_strength, diagnosis, diagnosis_3cat,
                                            current_age, gender, weight,
                                            education, ethnicity, race, apoe4, adas13, mmse, faq, 1])


                    elapsed_time = time.time() - start_time
                    logging.info('Elapsed time: %.2f secs' % elapsed_time)


                if not image_exists and INCLUDE_MISSING_IMAGES_IN_TABLE and write_csv:
                    # If the include missing images constant is set to true it will write all the rows to the table

                    csvwriter.writerow([rid, phase, image_exists, site, viscode, exam_date, field_strength, diagnosis, diagnosis_3cat,
                                        current_age, gender, weight,
                                        education, ethnicity, race, apoe4, adas13, mmse, faq, 1])






if __name__ == '__main__':

    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI1_screening_noPP/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI1_screening_reorient_crop_strip/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI1_screening_reorient_crop/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI1_screening_reorient_crop_strip_mni/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_ender_selection_reorient_crop/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_ender_selection_noPP/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_ender_selection_allPP_robex/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_all_no_skullstrip/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_all_no_PP_2/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_all_no_skullstrip/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_all_allPP_robex/')

    ### ----------

    processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_all_no_PP_3')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_allfixed_no_skullstrip/')
    # processed_images_folder = os.path.join(bmicdatasets_root, 'bmicdatasets/Processed/ADNI_Christian/ADNI_allfixed_allPP_robex/')

    utils.makefolder(processed_images_folder)
    summary_csv_file = os.path.join(processed_images_folder, 'summary_alldata.csv')

    do_reorientation = True  #True
    do_registration = True  #True
    do_bias_correction = True
    do_cropping = True #True
    do_skull_stripping = False #True

    # adnimerge_table = pd.read_csv(adni_merge_path, nrows=2)
    # adnimerge_table = pd.read_csv(adni_merge_path, chunksize=100)

    pool = multiprocessing.Pool(1)

    start_time = time.time()

    # func_list = []
    # for tmp_index, df in enumerate(adnimerge_table):
    #
    #     f = pool.apply_async(do_preprocessing, args=(df, tmp_index, processed_images_folder, summary_csv_file),
    #                                            kwds={'do_reorientation': do_reorientation,
    #                                                  'do_registration': do_registration,
    #                                                  'do_bias_correction': do_bias_correction,
    #                                                  'do_cropping': do_cropping,
    #                                                  'do_skull_stripping': do_skull_stripping,
    #                                                  'write_csv': True})
    #
    #     func_list.append(f)
    #
    #
    # for f in func_list:
    #     f.get()

    adnimerge_table = pd.read_csv(adni_merge_path)
    do_preprocessing(adnimerge_table, 0,
                     processed_images_folder,
                     summary_csv_file,
                     do_reorientation=do_reorientation,
                     do_registration=do_registration,
                     do_bias_correction=do_bias_correction,
                     do_cropping=do_cropping,
                     do_skull_stripping=do_skull_stripping,
                     write_csv=True)

    logging.info('Elapsed time %f secs' % (time.time()-start_time))
