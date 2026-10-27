#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using std::vector;

namespace Barenn {
class DenseLayer {

    int in_features;
    int out_features;
    int final;
    vector<vector<float>> weights;
    vector<float> bias;
    vector<float> last_input;
    vector<float> last_z;

    float relu(float sum) { return sum > 0.0 ? sum : 0.0; }

    float d_relu(float sum) { return sum > 0.0 ? 1.0 : 0.0; }

  public:
    DenseLayer(int inputSize, int outputSize, bool is_final = false) {
        in_features = inputSize;
        out_features = outputSize;
        final = is_final;
        weights = vector<vector<float>>(outputSize, vector<float>(inputSize));
        bias = vector<float>(outputSize, 0);
        std::mt19937 gen(1337);
        std::uniform_real_distribution<float> randDist(-0.05, 0.05);
        for (auto &p : weights) {
            for (auto &w : p) {
                w = randDist(gen);
            }
        }
    };

    vector<float> Forward(vector<float> input) {
        last_input = input;
        last_z.resize(out_features);
        vector<float> out(out_features, 0);
        for (int i = 0; i < out_features; i++) {
            float sum = bias[i];
            for (int j = 0; j < in_features; j++) {
                sum += input[j] * weights[i][j];
            }
            last_z[i] = sum;
            if(!final){
                out[i] = relu(sum);
            }
            else {
                out[i] = sum;
            }
        }
        return out;
    }

    vector<float> Backward(vector<float> grad_out, float learningrate) {
        vector<float> delta(out_features);
        for (int i = 0; i < out_features; i++) {
            if(!final){
                delta[i] = grad_out[i] * d_relu(last_z[i]);
            }
            else{
                delta[i] = grad_out[i];
            }
        }

        vector<float> grad_in(in_features, 0.0);
        for (int i = 0; i < in_features; i++) {
            float sum = 0;
            for (int j = 0; j < out_features; j++) {
                sum += delta[j] * weights[j][i];
            }
            grad_in[i] = sum;
        }

        for (int i = 0; i < out_features; i++) {
            for (int j = 0; j < in_features; j++) {
                weights[i][j] -= learningrate * (delta[i] * last_input[j]);
            }
            bias[i] -= learningrate * delta[i];
        }
        return grad_in;
    }
};

} // namespace Barenn

int main(int argc, char **argv) {

    // Lendo o traning set
    std::vector<std::vector<float>> trainingset{};
    std::vector<float> tag{};

    std::string rawline;
    std::ifstream testFile("optdigits.tra2");

    while (std::getline(testFile, rawline)) {
        std::stringstream streamline(rawline);
        std::string svalue;
        std::vector<float> image{};
        char delimit = ',';
        while (std::getline(streamline, svalue, delimit)) {
            float value = std::stof(svalue)/16;
            image.push_back(value);
        }
        tag.push_back(image.back() * 16);
        image.pop_back();
        trainingset.push_back(image);
    }
    // Fim da leitura
    //
    float learning_rate = 0.02;

    Barenn::DenseLayer l1(64, 32);
    Barenn::DenseLayer l2(32, 16);
    Barenn::DenseLayer l3(16, 10, true);

    for (int steps = 0; steps <= 50; steps++) {
        for (int i = 0; i < trainingset.size(); i++) {

            vector<float> image = trainingset[i];

            // Forward Pass
            vector<float> out1 = l1.Forward(image);
            vector<float> out2 = l2.Forward(out1);
            vector<float> pred = l3.Forward(out2);

            // MSE
            float loss = 0;
            vector<float> grad_loss(10);
            vector<float> target(10, 0);
            target[tag[i]] = 1.0;
            for (int j = 0; j < 10; j++) {
                float err = pred[j] - target[j];
                loss += 0.5 * std::pow(err, 2);
                grad_loss[j] = err;
            }

            // Backward Pass
            vector<float> bout3 = l3.Backward(grad_loss, learning_rate);
            vector<float> bout2 = l2.Backward(bout3, learning_rate);
            vector<float> bout1 = l1.Backward(bout2, learning_rate);

            if(steps == 50){
              std::cout << tag[i] << " - ";
              for (auto n : pred) {
                  std::cout << n << " ";
              }
              std::cout << std::endl;
            }
        }
    }
    return 0;
}
