import cv2
import glob
import numpy as np

query_image = input("Enter the image number: ")
for i in range(1, 7):
    fpath = 'testImage/' + str(i) + '/'
    path = glob.glob(fpath + '*.png')
    count = 1

    for image in path:
        query_img = cv2.imread('quaryImage/' + query_image + '.png')
        train_img = cv2.imread(image)

        query_img = cv2.resize(query_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        train_img = cv2.resize(train_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('img', query_img)

        query_img_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        train_img_gray = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

        # Initialize the ORB detector algorithm
        orb = cv2.ORB_create(1000)

        # Detect keypoints (features) and calculate the descriptors
        query_keypoints, query_descriptors = orb.detectAndCompute(query_img_gray, None)
        train_keypoints, train_descriptors = orb.detectAndCompute(train_img_gray, None)

        # Check if descriptors are not None
        if query_descriptors is not None and train_descriptors is not None:
            # Create a Brute-Force Matcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors
            matches = bf.match(query_descriptors.astype(np.uint8), train_descriptors.astype(np.uint8))

            # Sort them in ascending order of distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Calculate the number of matched keypoints
            matchingPer = len(matches) / len(query_keypoints) * 100

            if matchingPer > 70:
                print(f'Matched image {i} - Test image {count}')
            else:
                print(f'Not matched image {i} - Test image {count}')
        else:
            print(f'No descriptors found for image {i} - Test image {count}')

        count += 1

cv2.destroyAllWindows()
