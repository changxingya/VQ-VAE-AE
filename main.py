import cv2
import os

fps = 15
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter(filename='./result.avi', fourcc=fourcc, fps=fps, frameSize=(360, 240))

for k in range(198):

    root_img = '/Users/changxingya/Documents/Dataset/UCSD_flow/UCSDped1/Test/Test014/'+ str(k+1).zfill(3)+'.tif'
    root_gt = '/Users/changxingya/Desktop/论文工作/第一篇/CAE/pre_label/14/pre_label_'+ str(k)+'.png'

    img = cv2.imread(root_img)
    gt = cv2.imread(root_gt)

    print(root_gt)

    img = cv2.resize(img, (360, 240))

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j, 1] > 0:
                gt[i, j, 1] = 255
                gt[i, j, 0] = 0
                gt[i, j, 2] = 0

    dst = cv2.addWeighted(img, 0.7, gt, 0.3, 0)
    # cv2.waitKey(1000)
    # cv2.imwrite("./img/{}.png".format(k), dst)
    video_writer.write(dst)
video_writer.release()


