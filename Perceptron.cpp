#include <iostream>
#include <vector>

struct Table {
    bool x1;
    bool x2;
    bool answer;
};

bool Predict(
    std::int32_t biasInput, float biasedWeight,
    std::int32_t input1, float weight1,
    std::int32_t input2 = 0, float weight2 = 0.f,
    bool secondInput = true
) noexcept {
    float result = biasInput * biasedWeight + input1 * weight1;
    if (secondInput)
        result += input2 * weight2;

    return result > 0.f;
}

float GetWeightDelta(
    float learningRate, float input, std::int32_t error
) noexcept {
    return learningRate * error * input;
}

void Perceptron(
    float learningRate, const std::vector<Table>& dataTable,
    std::int32_t biasInput, float biasedWeight,
    float weight1,
    float weight2 = 0.f,
    bool secondInput = true
) noexcept {
    float errorCheck = 0.f;
    std::uint32_t count = 0u;
    do {
        float totalError = 0.f;
        for (const Table& data : dataTable) {
            bool calculatedResult = Predict(
                biasInput, biasedWeight,
                data.x1, weight1,
                data.x2, weight2,
                secondInput
            );

            std::int32_t error = data.answer - calculatedResult;
            totalError += error * error;

            std::cout << "\nInput1 = " << data.x1 << " Weight1 = " << weight1;

            if(secondInput)
                std::cout << "\nInput2 = " << data.x2 << " Weight2 = " << weight2;

            std::cout << "\nActual Result " << data.answer
                << " Calculated Result " << calculatedResult << "\n";

            if (error) {
                biasedWeight += GetWeightDelta(learningRate, biasInput, error);
                weight1 += GetWeightDelta(learningRate, data.x1, error);
                if (secondInput)
                    weight2 += GetWeightDelta(learningRate, data.x2, error);
            }
        }
        std::cout << "\nIteration " << ++count << " Error = " << totalError << "\n";
        errorCheck = totalError;
    } while (errorCheck >= 0.00001f);
}

int main() {
    std::int32_t biasInput = -1;
    const float learningRate = 0.25f;
    float biasedWeight = 0.3f;
    float weight1 = 0.5f;
    float weight2 = -0.4f;

    const std::vector<Table> andGate = {
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 1}
    };

    const std::vector<Table> orGate = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

    const std::vector<Table> nandGate = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}
    };

    const std::vector<Table> norGate = {
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 0}
    };

    // Perceptron(
    //     learningRate, nandGate,
    //     biasInput, biasedWeight,
    //     weight1, weight2
    // );

    const std::vector<Table> notGate = {
        {0, 0, 1},
        {1, 0, 0}
    };
    biasInput = 1;
    biasedWeight = 0.5f;
    weight1 = -1;

    Perceptron(
        learningRate, notGate,
        biasInput, biasedWeight,
        weight1, 0.f, false
    );

    return 0;
}