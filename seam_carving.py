import numpy as np
import cv2

class SeamCarvingClass:
    def __init__(self, filename, ratio_height, ratio_width):
        self.filename = filename
        self.in_image = cv2.imread(filename).astype(np.float64)
        cv2.imshow('Input',cv2.imread(filename))
        self.in_height, self.in_width = self.in_image.shape[: 2]
        self.out_height = int(self.in_height*ratio_height)
        self.out_width = int(self.in_width*ratio_width)
        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)
        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        self.seams_carving()
        self.save_result(output_image)
    def seams_carving(self):
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)
        # remove column
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)
        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)
####################################
    def seams_removal(self, num_pixel):
        for i in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)
###################################
    def seams_insertion(self, num_pixel):
        temp_image = np.copy(self.out_image)
        seams_record = []

        for i in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            seams_record.append(seam_idx)
            self.delete_seam(seam_idx)

        self.out_image = np.copy(temp_image)
        n = len(seams_record)
        for i in range(n):
            seam = seams_record.pop(0)
            self.add_seam(seam)
            seams_record = self.update_seams(seams_record, seam)
#############################
    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy
###########################
    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output
##################################
    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output
##################################
    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output
###########################
    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)
#############################
    def add_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.out_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)
##############################
    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output
#############################
    def rotate_image(self, image, ccw):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return output
######################################
    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
        cv2.imshow('Result',self.out_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
########################################

ratio_width = 0.8
ratio_height = 1.2

input_image = 'input8.jpg'
output_image = input_image[:-4]+'_result.jpg'
seam=SeamCarvingClass(input_image,ratio_height, ratio_width)



