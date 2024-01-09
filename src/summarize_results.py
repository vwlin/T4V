# parse LiRPA Verify output
# print statistics for unsafe images
# parameters
#   1. comma-separated list of n log files, representing n seeded trials of an experiment

import re
import numpy as np
from sys import argv

if __name__ == '__main__':
    paths = argv[1].split(',')
    num_trials = len(paths)

    ave_num_correct = 0
    ave_num_unsafe = 0
    ave_num_safe = 0
    ave_num_timeout = 0
    ave_percent_completed = 0
    ave_mn_time = 0
    ave_max_time = 0
    ave_min_time = 0

    for filename in paths:
        with open(filename, 'r') as file:
            log = file.read()

        new_img_indices = re.finditer('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n', log)
        new_img_indices = list(new_img_indices)

        num_img_tested = 0
        num_img_timeouts = 0
        
        num_unsafe_properties_per_img = [0]*len(tuple(new_img_indices))
        num_timeout_properties_per_img = [0]*len(tuple(new_img_indices))
        total_times = []

        total_splits_per_img = [0]*len(tuple(new_img_indices))

        for i in range(len(new_img_indices)):
            # if log contains results for more than 200 images, read results for only images 0-199
            if i == 200:
                break

            # proceed with parsing
            if i == len(new_img_indices)-1:
                correct = len(re.findall('model prediction is incorrect for the given model', log[new_img_indices[i].start():])) == 0
                completed = len(re.findall('did not time out', log[new_img_indices[i].start():])) > 0
            else:
                correct = len(re.findall('model prediction is incorrect for the given model', log[new_img_indices[i].start():new_img_indices[i+1].start()])) == 0
                completed = len(re.findall('did not time out', log[new_img_indices[i].start():new_img_indices[i+1].start()])) > 0

            if not correct:
                continue

            num_img_tested += 1

            # find out how many splits were made for each image
            if i == len(new_img_indices)-1:
                num_splits_per_img = list(re.finditer('# of unstable neurons: ', log[new_img_indices[i].start():]))
            else:
                splits = list(re.finditer('# of unstable neurons: ', log[new_img_indices[i].start():new_img_indices[i+1].start()]))
                if len(splits) > 0:
                    for split in splits:
                        split_start = new_img_indices[i].start() + splits[0].end()
                        split_end = split_start + log[split_start:].find('\n')
                        total_splits_per_img[i] += int(log[split_start : split_end])

            if not completed:
                num_img_timeouts += 1
                continue

            if i == len(new_img_indices)-1:
                num_unsafe_properties_per_img[i] = len(re.findall('UNSAFE', log[new_img_indices[i].start():]))
                num_timeout_properties_per_img[i] = len(re.findall('TIMEOUT', log[new_img_indices[i].start():]))
                
                time_start = list(re.finditer('total time: ', log[new_img_indices[i].start():]))
            else:
                num_unsafe_properties_per_img[i] = len(re.findall('UNSAFE', log[new_img_indices[i].start():new_img_indices[i+1].start()]))
                num_timeout_properties_per_img[i] = len(re.findall('TIMEOUT', log[new_img_indices[i].start():new_img_indices[i+1].start()]))
                
                time_start = list(re.finditer('total time: ', log[new_img_indices[i].start():new_img_indices[i+1].start()]))

            time_start = new_img_indices[i].start()+time_start[0].end()
            total_times.append( round(float(log[time_start : time_start + 8] ), 4) )

        if(num_img_tested == num_img_timeouts):
            print('No images were completed for:', filename)
            ave_num_correct += num_img_tested
            print('Number of images classified correctly:', num_img_tested)
            print()
            continue
       
        num_unsafe = sum(j > 0 for j in num_unsafe_properties_per_img)
        num_timeout = sum(j > 0 for j in num_timeout_properties_per_img) + num_img_timeouts
        num_safe = num_img_tested - num_timeout - num_unsafe
        percent_completed = (num_img_tested - num_timeout) / num_img_tested
        
        ave_num_correct += num_img_tested
        ave_num_unsafe += num_unsafe
        ave_num_safe += num_safe
        ave_num_timeout += num_timeout
        ave_percent_completed += percent_completed
        ave_mn_time += np.mean(total_times)
        ave_max_time += np.max(total_times)
        ave_min_time += np.min(total_times)
        #TODO add splits averages

        print('Results for:', filename)
        print('Number of images classified correctly:', num_img_tested)
        print('Total number of unsafe images:', num_unsafe)
        print('Total number of safe images:', num_safe)
        print('Total number of timeouts:', num_timeout)
        print('Percent of images completed:', percent_completed)
        print('Average time to verify all 9 properties:', np.mean(total_times))
        print('Max time to verify all 9 properties:', np.max(total_times))
        print()

    if(ave_percent_completed == 0):
        print('--------- AVERAGES ACROSS', str(num_trials), 'MODELS:', argv[1], '---------')
        print('No images were completed for any model verified')
        ave_num_correct /= int(num_trials)
        print('Number of images classified correctly:', ave_num_correct)
    else:
        numseeds = int(num_trials)
        ave_num_correct /= numseeds
        ave_num_unsafe /= numseeds
        ave_num_safe /= numseeds
        ave_num_timeout /= numseeds
        ave_percent_completed /= numseeds
        ave_mn_time /= numseeds
        ave_max_time /= numseeds
        ave_min_time /= numseeds
        #TODO add splits averages
        
        print('--------- AVERAGES ACROSS', str(numseeds), 'MODELS:', argv[1], '---------')
        print('Number of images classified correctly:', ave_num_correct)
        print('Total number of unsafe images:', ave_num_unsafe)
        print('Total number of safe images:', ave_num_safe)
        print('Total number of timeouts:', ave_num_timeout)
        print('Percent of images completed:', ave_percent_completed)
        print('Average time to verify all 9 properties:', ave_mn_time)
        print('Max time to verify all 9 properties:', ave_max_time)
        print('Min time to verify all 9 properties:', ave_min_time)
        #TODO add splits averages
        print()