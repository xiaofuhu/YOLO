from pydarknet import Detector, Image
import cv2
import os
import time
import json


def detect_car_only(net, image):
    image2 = Image(image)
    r = net.detect(image2)
    print('chck', r)
    return_list = []
    for item in r:
        #print(item)
        if item[0]==b'car' or item[0]==b'truck':
            return_list.append(item)
    return return_list
def measure_sim(item1, item2):
    item1_area = (item1['xright']-item1['xleft']) * (item1['ybottom']-item1['ytop'])
    item2_area = (item2['xright']-item2['xleft']) * (item2['ybottom']-item2['ytop'])
    print(item1_area, item2_area)
    #CHANGE by FUHU: min_area to max_area
    max_area = item2_area if item1_area<item2_area else item1_area

    overlap_left = item1['xleft'] if item1['xleft']>item2['xleft'] else item2['xleft']
    overlap_right = item2['xright'] if item1['xright']>item2['xright'] else item1['xright']
    overlap_top = item1['ytop'] if item1['ytop']>item2['ytop'] else item2['ytop']
    overlap_bottom = item1['ybottom'] if item1['ybottom']<item2['ybottom'] else item2['ybottom']

    if overlap_left>overlap_right or overlap_top>overlap_bottom:
        return 0
    else:
        print((overlap_right-overlap_left)*(overlap_bottom-overlap_top)/max_area)
        return (overlap_right-overlap_left)*(overlap_bottom-overlap_top)/max_area


def detection_to_coordinate(item, time, width, height):
    to_return = {
        'xleft': (item[2][0]-item[2][2]/2)/width,
        'xright': (item[2][0]+item[2][2]/2)/width,
        'ytop': (item[2][1]-item[2][3]/2)/height,
        'ybottom': (item[2][1]+item[2][3]/2)/height,
        'time': time
    }
    return to_return


def load_and_detect_video(net, filename):
    scale = 8
    vehicle_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,255,255)]
    #print(filename)
    cap = cv2.VideoCapture(os.path.join("input",filename))
    width = int(cap.get(3))  # float
    height = int(cap.get(4)) # float
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    out = cv2.VideoWriter(os.path.join("output",filename),0x00000021, fps, (width,height))
    videoname = filename.split('.')[0]
    to_return = {}
    counter = 0
    frame_num = 1
    while(cap.isOpened()):
        ret, ori_frame = cap.read()
        if ret==True:
            frame = cv2.resize(ori_frame, (int(width/scale), int(height/scale)))
            #frame = cv2.flip(frame,0)
            # write the flipped frame
            #cv2.imshow('frame',frame)
            r = detect_car_only(net, frame)
            curr_time = frame_num/fps
            to_return = add_vehicles(r, to_return, curr_time, videoname, width/scale, height/scale)
            
            #CHANGE by FUHU: added an if block
            for idx, item in enumerate(to_return.keys()):
                last_bbox = to_return[item]['bbox'][len(to_return[item]['bbox'])-1]
                t1 = last_bbox['time']
                print("LBT: ", t1, ", CT: ", curr_time)
                if t1 == curr_time:
                    cv2.rectangle(ori_frame, (int(last_bbox['xleft']*width), int(last_bbox['ytop']*height)),
                    (int(last_bbox['xright']*width), int(last_bbox['ybottom']*height)),
                    vehicle_colors[idx%len(vehicle_colors)], 2)

            with open(os.path.join('output',videoname+'.json'), 'w') as outfile:
                json.dump(dict_to_array(to_return), outfile, indent=4)

            out.write(ori_frame)
            #cv2.imshow('frame',ori_frame)
            frame_num=frame_num+1
            if 0xFF == ord('q'):
                break
            counter = counter+1
            
            #if counter>10:
            #    break
        else:
            break

    # Release everything if job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("*** Finished ***")

def add_vehicles(r, to_return, time, videoname, width, height):
    if len(to_return.keys())==0:
        for item in r:
            to_return['vehicle_'+str(len(to_return.keys()))] = {
                'videoname': videoname,
                'bbox': [detection_to_coordinate(item, time, width, height)],
                'objectname': 'vehicle_'+str(len(to_return.keys())),
            }

    else:
        vehi_ids = []
        for i in range(len(to_return.keys())):
            vehi_ids.append(None)
        for idx, item in enumerate(r):
            max_sim = -1
            max_sim_id = None
            for vehicle in to_return:
                if len(to_return[vehicle]['bbox'])>0:
                    cur_sim = measure_sim(to_return[vehicle]['bbox'][len(to_return[vehicle]['bbox'])-1], detection_to_coordinate(item, time, width, height))
                    if cur_sim > max_sim:
                        max_sim = cur_sim 
                        max_sim_id = vehicle
            if max_sim ==0:
                print('vehicle_'+str(len(vehi_ids)))
                to_return['vehicle_'+str(len(vehi_ids))] = {
                    'videoname': videoname,
                    'bbox': [],
                    'objectname': 'vehicle_'+str(len(vehi_ids)),
                }
                vehi_ids.append(idx)
            else:
                vehi_ids[int(max_sim_id.split('_')[1])] = idx
        for idx, vehi_id in enumerate(vehi_ids):
            if vehi_id!=None:
                to_return['vehicle_'+str(idx)]['bbox'].append(detection_to_coordinate(r[vehi_ids[idx]], time, width, height))
                #to_return[max_sim_id].append(item)
    return to_return

def dict_to_array(dic):
    to_return = []
    for key in dic:
        to_return.append(dic[key])
    return to_return

if __name__ == "__main__":
    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))
    load_and_detect_video(net, 'video1.mp4')
