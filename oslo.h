//
// Created by Andrew Liang on 03/03/2023.
//

#ifndef COMPLEXITY_OSLO_H
#define COMPLEXITY_OSLO_H

#include <iostream>
#include <vector>
#include <random>

class OsloModel {
public:
    explicit OsloModel(int L, int z_threshold_max = 2) :
            L(L),
            z_threshold_max(z_threshold_max+1),
            heights(L, 0),
            z(L, 0),
            z_threshold(L),
            time(0) {
        // initialize z_threshold randomly between 1 and z_threshold_max
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(1, z_threshold_max);
        for (int i = 0; i < L; i++) {
            z_threshold[i] = dist(gen);
        }
    }

    std::vector<int> heights;
    std::vector<int> z;
    std::vector<int> z_threshold;

    int run() {
        int s = 0;
        heights[0]++;
        z[0]++;

        while (!is_system_stable()) {
            if (z[0] > z_threshold[0]) {
                z[0] -= 2;
                z[1] += 1;
                heights[0] -= 1;
                heights[1] += 1;
                z_threshold[0] = get_new_z_threshold();
                s++;
            }
            for (int i = 1; i < L - 1; i++) {
                if (z[i] > z_threshold[i]) {
                    z[i] -= 2;
                    z[i + 1] += 1;
                    z[i - 1] += 1;
                    heights[i] -= 1;
                    heights[i + 1] += 1;
                    z_threshold[i] = get_new_z_threshold();
                    s++;
                }
            }
            if (z[L - 1] > z_threshold[L - 1]) {
                z[L - 1] -= 1;
                z[L - 2] += 1;
                heights[L - 1] -= 1;
                z_threshold[L - 1] = get_new_z_threshold();
                s++;
            }
            time++;
        }
        return s;
    }


private:
    int L;
    int z_threshold_max;
    int time;

    int get_new_z_threshold() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(1, z_threshold_max);
        return dist(gen);
    }

    bool is_system_stable() {
        for (int i = 0; i < L; i++) {
            if (z[i] > z_threshold[i]) {
                return false;
            }
        }
        return true;
    }
};

#endif //COMPLEXITY_OSLO_H
