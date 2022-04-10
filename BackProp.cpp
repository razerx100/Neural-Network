#include <iostream>
#include <cmath>

constexpr float E = 2.7182818284f;

struct IOs {
    const float input1;
    const float input2;
    const float biasInput1;
    const float biasInput2;
    const float output1;
    const float output2;
};

struct Weights {
    float weight1;
    float weight2;
    float weight3;
    float weight4;
    float weight5;
    float weight6;
    float weight7;
    float weight8;
    float biasWeight1;
    float biasWeight2;
};

struct CalculatedResults {
    float netH1;
    float outputH1;
    float netH2;
    float outputH2;
    float netO1;
    float outputO1;
    float netO2;
    float outputO2;
    float totalError;
};

std::ostream& operator<<(std::ostream& out, const Weights& weights) noexcept {
    return out << "W1 = " << weights.weight1 << " W2 = " << weights.weight2 << "\n"
        << "W3 = " << weights.weight3 << " W4 = " << weights.weight4 << "\n"
        << "W5 = " << weights.weight5 << " W6 = " << weights.weight6 << "\n"
        << "W7 = " << weights.weight7 << " W8 = " << weights.weight8;
}

constexpr float CalculateWeight(
    const float input1, const float weight1,
    const float input2, const float weight2,
    const float biasInput, const float biasWeight
) noexcept {
    return weight1 * input1 + weight2 * input2 + biasWeight * biasInput;
}

constexpr float LogisticFunction(
    const float netValue
) noexcept {
    return 1 / (1 + std::pow(E, -1 * netValue));
}

constexpr float LogisticFunctionBiPolar(
    const float netValue
) noexcept {
    return (1 - std::pow(E, -1 * netValue)) / (-1 + std::pow(E, -1 * netValue));
}

constexpr float CalculateError(
    const float targetOutput,
    const float calculatedOutput
) noexcept {
    return 0.5f * std::pow((targetOutput - calculatedOutput), 2.f);
}

constexpr float CalculatePartialOutput(
    const float targetOutput, const float calculatedOutput
) noexcept {
    return -(targetOutput - calculatedOutput);
}

constexpr float CalculatePartialNetValue(
    const float outputValue
) noexcept {
    return outputValue * (1 - outputValue);
}

constexpr CalculatedResults ForwardPass(const IOs& io, const Weights& weights) noexcept {
    const float netH1 = CalculateWeight(
        io.input1, weights.weight1, io.input2,
        weights.weight2, io.biasInput1, weights.biasWeight1
    );

    const float outputH1 = LogisticFunction(netH1);

    const float netH2 = CalculateWeight(
        io.input1, weights.weight3, io.input2,
        weights.weight4, io.biasInput1, weights.biasWeight1
    );

    const float outputH2 = LogisticFunction(netH2);

    const float netO1 = CalculateWeight(
        outputH1, weights.weight5, outputH2,
        weights.weight6, io.biasInput2, weights.biasWeight2
    );

    const float outputO1 = LogisticFunction(netO1);

    const float netO2 = CalculateWeight(
        outputH1, weights.weight7, outputH2,
        weights.weight8, io.biasInput2, weights.biasWeight2
    );

    const float outputO2 = LogisticFunction(netO2);

    const float error1 = CalculateError(io.output1, outputO1);

    const float error2 = CalculateError(io.output2, outputO2);

    const float totalError = error1 + error2;

    CalculatedResults result = {
        netH1, outputH1,
        netH2, outputH2,
        netO1, outputO1,
        netO2, outputO2,
        totalError
    };

    return result;
}

constexpr float CalculateOutputWeight(
    const float partialNet, const float output,
    const float learningRate, const float weight
) noexcept {
    const float partialTotal = partialNet * output;

    return weight - learningRate * partialTotal;
}

constexpr Weights BackwardPass(
    const IOs& io, const Weights& weights,
    const CalculatedResults& result,
    const float learningRate
) noexcept {
    // Net O1    
    const float partialNetO1 = CalculatePartialOutput(io.output1, result.outputO1)
        * CalculatePartialNetValue(result.outputO1);

    const float newWeight5 = CalculateOutputWeight(
        partialNetO1, result.outputH1, learningRate, weights.weight5
    );

    const float newWeight6 = CalculateOutputWeight(
        partialNetO1, result.outputH2, learningRate, weights.weight6
    );

    // Net O2
    const float partialNetO2 = CalculatePartialOutput(io.output2, result.outputO2)
        * CalculatePartialNetValue(result.outputO2);

    const float newWeight7 = CalculateOutputWeight(
        partialNetO2, result.outputH1, learningRate, weights.weight7
    );

    const float newWeight8 = CalculateOutputWeight(
        partialNetO2, result.outputH2, learningRate, weights.weight8
    );

    // Hidden Layer

    // Net H1
    const float partialOutH1W5 = partialNetO1 * weights.weight5;

    const float partialOutH1W7 = partialNetO2 * weights.weight7;

    const float partialOutH1 = partialOutH1W5 + partialOutH1W7;

    const float partialNetH1 = partialOutH1 * CalculatePartialNetValue(result.outputH1);

    // Net H2
    const float partialOutH2W6 = partialNetO1 * weights.weight6;

    const float partialOutH2W8 = partialNetO2 * weights.weight8;

    const float partialOutH2 = partialOutH2W6 + partialOutH2W8;

    const float partialNetH2 = partialOutH2 * CalculatePartialNetValue(result.outputH2);

    // Weights
    const float newWeight1 = CalculateOutputWeight(
        partialNetH1, io.input1,
        learningRate, weights.weight1
    );

    const float newWeight2 = CalculateOutputWeight(
        partialNetH1, io.input2,
        learningRate, weights.weight2
    );

    const float newWeight3 = CalculateOutputWeight(
        partialNetH2, io.input1,
        learningRate, weights.weight3
    );

    const float newWeight4 = CalculateOutputWeight(
        partialNetH2, io.input2,
        learningRate, weights.weight4
    );

    Weights newWeights = {
        newWeight1, newWeight2, newWeight3, newWeight4,
        newWeight5, newWeight6, newWeight7, newWeight8,
        weights.biasWeight1, weights.biasWeight2
    };

    return newWeights;
}

void BackPropagate(
    const IOs& io, const Weights& weights,
    float learningRate, size_t count
) noexcept {
    Weights newWeight = weights;

    for (size_t index = 1u; index <= count; ++index) {
        CalculatedResults forwardResult = ForwardPass(io, newWeight);
        newWeight = BackwardPass(io, newWeight, forwardResult, learningRate);

        std::cout << "===== Iteration " << index << " =====\n";
        std::cout << "Total Error = " << forwardResult.totalError << "\n";
        std::cout << newWeight << "\n";
    }
}

int main() {
    constexpr IOs io = { 0.05f, 0.10f, 1.f, 1.f, 0.01f, 0.99f };
    constexpr Weights weights = { 0.15f, 0.20f, 0.25f, 0.30f, 0.40f,
    0.45f, 0.50f, 0.55f, 0.35f, 0.60f };

    BackPropagate(io, weights, 0.5f, 2u);

    return 0;
}