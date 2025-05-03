import numpy as np
import pickle as pk
import multiprocessing
import time
import os
import pandas as pd


class radarConfig():
    def __init__(self):
        self.NUM_TX = 3
        self.NUM_TX_LOWER = 2
        self.NUM_RX = 4
        self.NUM_TRX = self.NUM_TX * self.NUM_RX
        self.CHIRP_LOOPS = 128
        self.ADC_SAMPLES = 256
        self.NUM_RANGE_BINS = self.ADC_SAMPLES
        self.NUM_DOPPLER_BINS = self.CHIRP_LOOPS

        self.FRAME_SIZE = self.NUM_TX * self.NUM_RX * self.CHIRP_LOOPS * self.ADC_SAMPLES

        self.FRAME_BYTE_SIZE = self.FRAME_SIZE * 4
        self.FRAME_INT16_SIZE = self.FRAME_SIZE * 2
        self.FRAME_COMPLEX_SIZE = self.FRAME_SIZE

        self.ANGLE_BIN_SIZE = 128
        self.FPS = 10


class transform():
    def __init__(self):
        self.bin_location = './bin_files/'
        self.out_root_loc = './raw_radar_frames/'
        if not os.path.isdir(self.out_root_loc):
            os.mkdir(self.out_root_loc)
        self.skip_sec_from_begin = 0
        self.sync_list = self._load_sync_list('./sync_file.csv')
        self.cfg = radarConfig()


    def save_np_mat(self, mat, name):
        np.save(name, mat) # save mat array into the file (name)

    # def _load_sync_list(self, infilename):
    #     sync_list = []
    #     with open(infilename) as infile:
    #         print(infilename)
    #         for line in infile.readlines():
    #             file_name=line.split(" ")[0]
    #             start_frame = line.split(" ")[-2]
    #             trail_name= line.split(" ")[-1]
    #             sync_list.append([file_name, int(start_frame), trail_name.replace("\n","")])
    #     return sync_list

    def _load_sync_list(self, infilename):
        sync_list = []
        csv_file = pd.read_csv(infilename, header=0)
        print(infilename)
        for row_index in range(csv_file.shape[0]):
            file_name = csv_file["bin_name"][row_index] + ".bin"
            start_frame = csv_file["radar_start_frame"][row_index]
            trail_name = csv_file["trial_name"][row_index]
            sync_list.append([file_name, int(start_frame), trail_name])
        print(sync_list)
        return sync_list

    def load_binary_frames(self, infilename, batch_length, skip_size): #batch length = num frame
        frame_byte_size = self.cfg.FRAME_BYTE_SIZE
        frame_int16_size = self.cfg.FRAME_INT16_SIZE
        frame_complex_size = self.cfg.FRAME_COMPLEX_SIZE
        with open(infilename, 'rb') as infile_bin: #open binary file in read mode
            infile_bin.seek(skip_size * frame_byte_size, 0) #frame byte size
            frame_read = np.frombuffer(infile_bin.read(batch_length * frame_byte_size), dtype=np.int16)#How many bytes to read, read out as two bytes per int16
        assert len(frame_read) == frame_int16_size * batch_length
        frame_out = np.zeros(shape=(batch_length * frame_complex_size,), dtype=np.complex128)
        frame_out[0::2] = frame_read[0::4] + 1j * frame_read[2::4]
        frame_out[1::2] = frame_read[1::4] + 1j * frame_read[3::4]
        #print(frame_out)
        frame_np = frame_out.reshape(batch_length, self.cfg.CHIRP_LOOPS, self.cfg.NUM_TX, self.cfg.NUM_RX,
                                     self.cfg.ADC_SAMPLES).transpose(0, 2, 3, 1, 4)  # (batch_length, 3, 4, 128, 256)
        return frame_np

    def get_processed_frames(self, infilename, frame_begin, frame_size, skip_size):
        for frame_no in range(frame_size):
            frames_np = self.load_binary_frames(infilename, 1, skip_size + frame_no).squeeze(0)  # (batch,3,4,128,256)
            with open(self.dump_folder + '%04d.npy' % (frame_begin + frame_no), 'wb') as outfile:
                np.save(outfile, frames_np.astype(np.csingle))

    def sprocessing_dump_sample(self, filename_idx, frame_begin, frame_size):
        infilename = self.bin_location + self.sync_list[filename_idx][0]
        skip_size = self.sync_list[filename_idx][1] + self.skip_sec_from_begin * self.cfg.FPS + frame_begin
        print('\t\t', frame_begin, infilename, skip_size)

        self.get_processed_frames(infilename, frame_begin, frame_size, skip_size)
        print('\t\tprocess %d + %d done' % (frame_begin, frame_size))

    def mprocessing_simulated_signal(self, filename_idx, total_length, process_size):
        self.keyword = self.sync_list[filename_idx][-1]
        self.dump_folder = self.out_root_loc + '%s/' % (self.keyword)
        if not os.path.isdir(self.dump_folder):
            os.mkdir(self.dump_folder)

        print('\tmultiprocessing begin')
        begin_time = time.time()
        process_list = []
        piece_size = total_length // process_size # round down
        if total_length % process_size != 0:
            process_size += 1
        print('\t\tprocess_size:', process_size)
        for i in range(process_size - 1):
            t = multiprocessing.Process(target=self.sprocessing_dump_sample,
                                        args=(filename_idx, i * piece_size, piece_size))
            t.start()
            process_list.append(t)
        # add the rest left part (mod part)
        i = process_size - 1
        t = multiprocessing.Process(target=self.sprocessing_dump_sample,
                                    args=(filename_idx, i * piece_size, total_length - piece_size * i))
        t.start()
        process_list.append(t)

        for t in process_list:
            t.join()
        end_time = time.time()
        print('\tmultiprocessing end: %.1f min(s)' % ((end_time - begin_time) / 60))


if __name__ == '__main__':
    t = transform()
    filename_idx_list = list(range(0,len(t.sync_list)))
    # filename_idx_list=[14]
    for filename_idx in filename_idx_list:
        print(filename_idx)
        t.mprocessing_simulated_signal(filename_idx, 50, 10)#
