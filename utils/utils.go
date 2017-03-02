package utils

import(
    
    "fmt"
    "math"
    "math/rand"
)

func Uniform(min float64, max float64) float64{
    
    return rand.Float64() * (max - min) + min
}

func Activation(activationName string, input float64) float64{

    var x float64
    if activationName == "ReLU"{
        
        if input >= 0.0{
            x = input
        }else{
            x = 0.0
        }
    }
    if activationName == "Sigmoid"{

        input *= -1
        x = 1.0 / (1 + math.Exp(input))
    }
    if activationName == "Tanh"{
        
        x = math.Tanh(input)
    }
    return x
}

func Dactivation(activationName string, input float64) float64{

    var x float64
    if activationName == "ReLU"{

        if input >= 0.0{
            x = 1.0
        }else{
            x = 0.0
        }
    }
    if activationName == "Sigmoid"{

        input *= -1
        x = (1 - 1.0 / (1 + math.Exp(input))) * (1.0 / (1 + math.Exp(input)))
    }
    if activationName == "Tanh"{
        
        x = 1 / math.Pow(math.Cosh(input), 2)
    }
    return x
}

func SoftMax(input []float64) []float64{

    var(
        max float64 = 0.0
        sum float64 = 0.0
    )

    size := len(input)
    var output  = make([]float64, size)

    for i := 0; i < size; i++{
        if input[i] > max{
            max = input[i]
        }
    }
    
    for i := 0; i < size; i++{
        output[i] = math.Exp(input[i] - max)
        sum += output[i]
    }
    
    for i := 0; i < size; i++{
        output[i] /= sum
    }
    return output
}

func CrossEntropy(input []float64, label []int) float64{

    var output float64
    
    size := len(input)
    for i := 0; i < size; i++{
        output += (-1) * math.Log(input[i]) * float64(label[i])
    }
    
    fmt.Println("CrossEntropy")
    fmt.Println(output)
    
    return output
}