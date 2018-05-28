'''Seperate image files for cv from all files in a folder to other folder.
'''

import os

FOLDER_CURRENT = os.getcwd()
FOLDER_FROM = 'image_train'
FOLDER_TO = 'image_cv'
categories = os.listdir(FOLDER_FROM)

for j in range(0, len(categories)):
    categoryContents = sorted(os.listdir(
        os.path.join(FOLDER_FROM, categories[j])))

    for i in range(0, len(categoryContents)):
        if i % 5 == 0:
            FILE_FROM = os.path.join(FOLDER_CURRENT, FOLDER_FROM,
                                     categories[j], categoryContents[i])
            FILE_TO = os.path.join(FOLDER_CURRENT, FOLDER_TO,
                                   categories[j], categoryContents[i])
            os.rename(FILE_FROM, FILE_TO)
