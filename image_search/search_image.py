import cv2
import os
import sys
import numpy as np
import pickle
import glob

def keypoint_to_tuple(kp):
    #print("Converting to tuple: %s" % str((kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)))
    return (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)

def tuple_to_keypoint(t):
    #print(f"Converting tuple: {t}")  # Debug print statement
    # Directly pass the first three parameters without names
    return cv2.KeyPoint(t[0][0], t[0][1], t[1], angle=t[2], response=t[3], octave=t[4], class_id=t[5])

def create_parent_folders_for_file(file_path):
    # Extract the directory part of the file path
    directory_path = os.path.dirname(file_path)
    
    # Recursively create all parent directories for the file
    # exist_ok=True means Python will not throw an error if the directory already exists
    os.makedirs(directory_path, exist_ok=True)

def cache_features(keypoint_nr, source_images_dir, cache_dir):
    orb = cv2.ORB_create(keypoint_nr)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    print("Caching features")

    for file in glob.glob(source_images_dir + "/**/*.jpg", recursive = True):
        #print("Caching %s" % file)
        cache_path = file.replace(source_images_dir, cache_dir).replace(".jpg", ".pkl")
        #print("Cache path: %s" % cache_path)
        
        create_parent_folders_for_file(cache_path)
        
        # Skip if cache already exists
        if os.path.exists(cache_path):
            continue
        
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        keypoints, des = orb.detectAndCompute(image, None)

        test = [tuple_to_keypoint(keypoint_to_tuple(kp)) for kp in keypoints]
        
        kp_tuples = [keypoint_to_tuple(kp) for kp in keypoints]
        
        # Cache keypoints and descriptors
        with open(cache_path, 'wb') as f:
            pickle.dump((kp_tuples, des), f)

def find_best_match_with_cache(keypoint_nr, target_image_path, source_images_dir, cache_dir):
    orb = cv2.ORB_create(keypoint_nr)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    kp_target, des_target = orb.detectAndCompute(target_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_img_path = None
    highest_num_of_matches = 0
    
    # First, get the list of all .pkl files to determine the total count
    cache_files = list(glob.glob(cache_dir + "/**/*.pkl", recursive=True))
    total_files = len(cache_files)
    percent_threshold = total_files / 100  # 1% of total files
    processed_files = 0

    for cache_file in cache_files:
        processed_files += 1
        with open(cache_file, 'rb') as f:
            kp_source, des_source = pickle.load(f)
            # Convert keypoints from serialized objects
            kp_source = [tuple_to_keypoint(t) for t in kp_source]

            if des_source is not None and des_target is not None:
                matches = bf.match(des_target, des_source)
                num_of_matches = len(matches)
                
                if num_of_matches > highest_num_of_matches:
                    highest_num_of_matches = num_of_matches
                    best_match_img_path = cache_file  # Remove .pkl extension

        if processed_files % percent_threshold < 1:
            print(f"Processed {processed_files / total_files * 100:.2f}% of images...")

    orig_path = best_match_img_path.replace(cache_dir, source_images_dir).replace(".pkl", ".jpg")
    return orig_path, highest_num_of_matches

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 search_image.py keypoint_nr /path/to/target/image")
        sys.exit(1)

    keypoint_nr = int(sys.argv[1])
    # Cache the features of source images
    source_images_dir = os.path.join(os.path.expanduser('~'), 'search_dir')
    cache_dir = os.path.join(os.path.expanduser('~'), 'cache_dir')
    cache_features(keypoint_nr, source_images_dir, cache_dir)

    # Find the best match using the cached features
    target_image_path = sys.argv[2]
    best_match, matches = find_best_match_with_cache(keypoint_nr, target_image_path, source_images_dir, cache_dir)
    if best_match:
        print(f"Best match: {best_match} with {matches} matches")
    else:
        print("No match found.")
