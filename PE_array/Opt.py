import numpy
import matplotlib.pyplot as plt

class MacModel:
    def __init__(self, DSP_tol=1024, alg_model=None, input_size=16):
        self.DSP_tol = DSP_tol
        self.cycle_time = 0
        self.perf_eff = 0
        self.band_width = 0
        self.tol_mem = 0
        self.alg_model = alg_model
        self.input_size = input_size

        self.dsp_eff = []
        self.core_num_list = []
        self.core_size_list = []
        self.bandwidth_list = []
        self.bandwidth_eff_list = []
        self.tol_cycle_time_list = []
        self.mean_perf_eff_list = []
        self.tol_buf_list = []
        self.pe_buf_list = []
        self.core_buf_list = []

    def cal_mem(self, core_num, core_size):
        pe_reg = self.input_size * 4 * (core_num * core_size ** 2)
        core_buf = core_size * (core_size * 2 - 1) * (2 * self.input_size)
        self.tol_mem = (pe_reg + core_buf) / 1024 / 8
        # print("Tol Buffer size is ", self.tol_mem, "KB", "; PE Buffer size is ", pe_reg / 1024 / 8, "KB", \
        #       "; Core Buffer size is ", core_buf / 1024 / 8, "KB")
        return pe_reg / 1024 / 8, core_buf / 1024 / 8

    def cal_cycle_time_and_perf_eff(self, size=None, core_size=16, max_bandwidth=2048):
        core_num = int(self.DSP_tol / (core_size ** 2))
        # print(self.DSP_tol, core_size, core_num)
        bandwidth = core_num * core_size * 16
        if bandwidth > max_bandwidth:
            return 0, max_bandwidth+1

        Rd2 = size[0] - int(size[0] / core_size) * core_size
        Rd3 = size[2] - int(size[2] / core_size) * core_size

        if int(size[1] / core_num) < size[1] / core_num:
            b = (int(size[1] / core_num) + 1)
        else:
            b = size[1] / core_num

        cycle_time = int(size[0] / core_size) * int(size[2] / core_size) * (2 * core_size + b) * core_num \
                     + int(size[0] / core_size) * (core_size + Rd2 + b) * core_num \
                     + int(size[2] / core_size) * (core_size + Rd3 + b) * core_num \
                     + (Rd2 + Rd3 + b) * core_num
        self.cycle_time = cycle_time / core_num
        self.perf_eff = size[0] * size[1] * size[2] * 3 / (3 * (core_size ** 2) * cycle_time)

        return core_num, bandwidth

    def run(self, max_size=32):
        global bw_eff
        for i in range(max_size):
            tol_cycle_time = 0
            mean_perf_eff = 0
            bandwidth = 0
            core_num = 0
            core_size = 0
            pe_buf = 0
            core_buf = 0
            bw_eff = 0
            for size in self.alg_model:
                core_size = i + 1
                core_num, bandwidth = self.cal_cycle_time_and_perf_eff(size=size, core_size=core_size)

                if bandwidth > 2048:
                    break
                tol_cycle_time += self.cycle_time
                mean_perf_eff += self.perf_eff

                pe_buf, core_buf = self.cal_mem(core_num=core_num, core_size=core_size)

                for j in range(12):
                    bw_eff = 0
                    if 2**j < bandwidth < 2**(j + 1):
                        bw_eff = bandwidth/(2**(j+1))
                        break
                    elif bandwidth==2**j:
                        bw_eff = 1
                        break

            if bandwidth > 2048:
                continue

            print("core num is", core_num, " size is", core_size, " bandwidth is", bandwidth, " bandwidth eff is",
                  bw_eff, " tol_cycle_time is",
                  tol_cycle_time, " mean_perf_eff is", mean_perf_eff / len(self.alg_model),
                  "Tol Buffer size is ", self.tol_mem, "KB", "; PE Buffer size is ", pe_buf, "KB",
                  "; Core Buffer size is ", core_buf, "KB")
            self.core_num_list.append(core_num)
            self.core_size_list.append(core_size)
            self.dsp_eff(core_size*core_num/self.DSP_tol)
            self.bandwidth_list.append(bandwidth)
            self.bandwidth_eff_list.append(bw_eff)
            self.tol_cycle_time_list.append(tol_cycle_time)
            self.mean_perf_eff_list.append(mean_perf_eff)
            self.tol_buf_list.append(self.tol_mem)
            self.pe_buf_list.append(pe_buf)
            self.core_buf_list.append(core_buf)


def main_test():
    seq_len = 100   # 100， 300， 500
    hidden_size = 768    # 768， 3072
    DSP_tol = 1024
    size_0 = [seq_len, hidden_size, hidden_size/12]  # a, b, c
    size_1 = [seq_len, hidden_size/12, seq_len]
    size_2 = [seq_len, seq_len, hidden_size/12]
    size_3 = [seq_len, 3096, hidden_size]
    size_4 = [seq_len, hidden_size, 3096]

    size_tol = [size_0, size_0, size_0, size_1, size_2, size_3, size_4]
    mac = MacModel(DSP_tol=DSP_tol, alg_model=size_tol)
    # print(len(mac.alg_model))
    mac.run()



def main():
    # size:
    # [n, 768, 64] * 3
    # [n, 64, n] * 1
    # [n, n, 64] * 1
    n = 200
    size_0 = [n, 768, 64]  # a, b, c
    size_1 = [n, 64, n]
    size_2 = [n, n, 64]
    size_3 = [n, 3096, 768]
    size_4 = [n, 768, 3096]
    DSP_tol = 1024

    tol_mem = (768 * 64 * 4 + n * 64 * 3 + n * n + n * 64 + n * 768 + 6 * 768) * 16 / 8 / 1024
    print("SRAM size is ", tol_mem, "KB")

    for i in range(32):
        tol_cycle_time = 0
        mean_perf_eff = 0
        for size in [size_0, size_0, size_0, size_1, size_2]:
            L = i + 1
            N = int(DSP_tol / (L ** 2))

            Rd2 = size[0] - int(size[0] / L) * L
            Rd3 = size[2] - int(size[2] / L) * L

            bandwidth = N * L * 16
            if bandwidth > 2048:
                continue

            if int(size[1] / N) < size[1] / N:
                b = (int(size[1] / N) + 1)
            else:
                b = size[1] / N

            cycle_time = int(size[0] / L) * int(size[2] / L) * (2 * L + b) * N \
                         + int(size[0] / L) * (L + Rd2 + b) * N \
                         + int(size[2] / L) * (L + Rd3 + b) * N \
                         + (Rd2 + Rd3 + b) * N

            cycle_time_min = int(size[0] / L) * int(size[2] / L) * (2 * L + b) * N \
                             + int(size[0] / L) * (b) * N \
                             + int(size[2] / L) * (b) * N \
                             + (b) * N

            cycle_time_mean = cycle_time / N
            tol_cycle_time += cycle_time_mean

            perf_eff = size[0] * size[1] * size[2] * 3 / (3 * (L ** 2) * cycle_time)
            mean_perf_eff += perf_eff

            # print("core num is", N, " size is", L, "bandwidth is", bandwidth, " cycletime is", cycle_time_mean,
            # " performance eff is", perf_eff)

        if bandwidth > 2048:
            continue

        memory_size = 32 * (L ** 2) * N + (L ** 2) * 16 * 2

        print("core num is", N, " size is", L, " bandwidth is", bandwidth, " tol_cycle_time is",
              tol_cycle_time, " mean_perf_eff is", mean_perf_eff / 5, " memory size is", memory_size)


if __name__ == "__main__":
    main_test()
