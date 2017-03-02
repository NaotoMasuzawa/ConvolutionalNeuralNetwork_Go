package ConvolveNeuralNetwork

import(
    
    "fmt"
    dirCPL     "../ConvolutionPoolingLayer"
    dirConnect "../Connect"
    dirMLP     "../MultiLayerPerceptron"
    dirLR      "../LogisticRegression"
)

type ConvolveNeuralNetwork struct{

    nConvPoolLayers int
    nMLP int
    class int

    CPL []dirCPL.ConvolutionalPoolingLayer
    Connect dirConnect.Connect
    MLP []dirMLP.MultiLayerPerceptron
    LR dirLR.LogisticRegression
}

func Construct(self *ConvolveNeuralNetwork,
               imageSize          []int,
               channel              int,
               nKernel            []int,
               kernelSizes      [][]int,
               poolSizes        [][]int,
               mlpSizes           []int,
               class                int,
               miniBatchSize        int,
               dropOut            []bool,
               dropOutPossibility []float64,
               activationName       string){

    self.nConvPoolLayers = len(nKernel)
    self.nMLP = len(mlpSizes)
    self.class = class

    fmt.Println("Construct the Convolve and Pooling layer.")
    self.CPL = make([]dirCPL.ConvolutionalPoolingLayer, self.nConvPoolLayers)

    var(
        inSize = make([][]int, self.nConvPoolLayers)
        convedSize = make([][]int, self.nConvPoolLayers)
        pooledSize = make([][]int, self.nConvPoolLayers)
    )
    for i := 0; i < self.nConvPoolLayers; i++{
        inSize[i] = make([]int, 2)
        convedSize[i] = make([]int, 2)
        pooledSize[i] = make([]int, 2)
    }
    
    for i := 0; i < self.nConvPoolLayers; i++{
        var eachChannel int

        if i == 0{
            inSize[i][0] = imageSize[0]
            inSize[i][1] = imageSize[1]
            eachChannel = channel
        }else{
            inSize[i][0] = pooledSize[i - 1][0]
            inSize[i][1] = pooledSize[i - 1][1]
            eachChannel = nKernel[i - 1]
        }
        convedSize[i][0] = inSize[i][0] - kernelSizes[i][0] + 1
        convedSize[i][1] = inSize[i][1] - kernelSizes[i][1] + 1
        pooledSize[i][0] = convedSize[i][0] / poolSizes[i][0]
        pooledSize[i][1] = convedSize[i][1] / poolSizes[i][1]

        fmt.Printf("Construct the %d layer.\n", i + 1)
        dirCPL.Construct(&(self.CPL[i]), inSize[i], eachChannel, nKernel[i], kernelSizes[i], poolSizes[i], convedSize[i], pooledSize[i], miniBatchSize, activationName)
        dirCPL.Confirm(&(self.CPL[i]))
    }
    
    fmt.Println("-----------------------------------")
    fmt.Println("Construct the Connection.")
    flattenedSize := nKernel[self.nConvPoolLayers - 1] * pooledSize[self.nConvPoolLayers - 1][0] * pooledSize[self.nConvPoolLayers - 1][1]
    dirConnect.Construct((&self.Connect), miniBatchSize, nKernel[self.nConvPoolLayers - 1], pooledSize[self.nConvPoolLayers - 1], flattenedSize, mlpSizes[0])

    fmt.Println("-----------------------------------")
    fmt.Println("Construct the MultiLayerPerceptron.")
    self.MLP = make([]dirMLP.MultiLayerPerceptron, self.nMLP)
    
    for i := 0; i < self.nMLP; i++{
        var in int

        if i == 0{
            in = flattenedSize
        }else{
            in = mlpSizes[i - 1]
        }

        fmt.Printf("Construct the %d layer.\n", i + 1)
        dirMLP.Construct((&self.MLP[i]), in, mlpSizes[i], dropOut[i], dropOutPossibility[i], miniBatchSize, activationName)
        dirMLP.Confirm((&self.MLP[i]))
    }

    fmt.Println("---------------------------------")
    fmt.Println("Construct the LogisticRegression.")
    dirLR.Construct((&self.LR), mlpSizes[self.nMLP - 1], class, miniBatchSize)
    dirLR.Confirm(&self.LR)
    fmt.Println("---------------------------------")
}

func Train(self *ConvolveNeuralNetwork, input [][][][]float64, actualLabel [][]int, learningRate float64){

    TrainOrTest := "Train"

    fmt.Println("ConvolutionPoolingLayer")
    for i := 0; i < self.nConvPoolLayers; i++{
        if i == 0{
            fmt.Printf("%d layer\n", i + 1)
            dirCPL.Forward((&self.CPL[i]), input, TrainOrTest)
        }else{
            fmt.Printf("%d layer\n", i + 1)
            dirCPL.Forward((&self.CPL[i]), self.CPL[i - 1].Pooled, TrainOrTest)
        } 
    }

    fmt.Println("Conneting")
    dirConnect.Flatten((&self.Connect), self.CPL[self.nConvPoolLayers - 1].Pooled)
    
    fmt.Println("MultiLayerPerceptron")
    for i := 0; i < self.nMLP; i++{
        if i == 0{
            fmt.Printf("%d layer\n", i + 1)
            dirMLP.Forward((&self.MLP[i]), self.Connect.Flattened, TrainOrTest)
        }else{
            fmt.Printf("%d layer\n", i + 1)
            dirMLP.Forward((&self.MLP[i]), self.MLP[i - 1].Activated, TrainOrTest)
        }
    }

    fmt.Println("LogisticRegression")
    dirLR.Train((&self.LR), self.MLP[self.nMLP - 1].Activated, actualLabel, learningRate)

    fmt.Println("Back MLP")
    for i := self.nMLP - 1; 0 <= i; i--{
        if i == self.nMLP - 1{
            fmt.Printf("%d layer\n", i + 1)
            dirMLP.Backward((&self.MLP[i]), self.LR.Delta, self.LR.W, self.class, learningRate)
        }else{
            fmt.Printf("$d layer\n", i + 1)
            dirMLP.Backward((&self.MLP[i]), self.MLP[i + 1].Delta, self.MLP[i + 1].W, self.MLP[i + 1].Out, learningRate)
        }
    }

    fmt.Println("Connecting")
    dirConnect.Unflatten((&self.Connect), self.MLP[0].Delta, self.MLP[0].W)

    fmt.Println("Back ConvPool layer")
    for i := self.nConvPoolLayers - 1; 0 <= i; i--{
        if i == self.nConvPoolLayers - 1{
            fmt.Printf("%d layer\n", i + 1)
            dirCPL.Backward((&self.CPL[i]), self.Connect.Unflattened, learningRate)
        }else{
            fmt.Printf("%d layer\n", i + 1)
            dirCPL.Backward((&self.CPL[i]), self.CPL[i + 1].DConvolve, learningRate)
        }
    }
}

func Test(self *ConvolveNeuralNetwork, input [][][][]float64, actualLabel [][]int){

    TrainOrTest := "Test"

    fmt.Println("ConvolutionPoolingLayer")
    for i := 0; i < self.nConvPoolLayers; i++{
        if i == 0{
            fmt.Printf("%d layer\n", i + 1)
            dirCPL.Forward((&self.CPL[i]), input, TrainOrTest)
        }else{
            fmt.Printf("%d layer\n", i + 1)
            dirCPL.Forward((&self.CPL[i]), self.CPL[i - 1].Pooled, TrainOrTest)
        } 
    }

    fmt.Println("Conneting")
    dirConnect.Flatten((&self.Connect), self.CPL[self.nConvPoolLayers - 1].Pooled)
    
    fmt.Println("MultiLayerPerceptron")
    for i := 0; i < self.nMLP; i++{
        if i == 0{
            fmt.Printf("%d layer\n", i + 1)
            dirMLP.Forward((&self.MLP[i]), self.Connect.Flattened, TrainOrTest)
        }else{
            fmt.Printf("%d layer\n", i + 1)
            dirMLP.Forward((&self.MLP[i]), self.MLP[i - 1].Activated, TrainOrTest)
        }
    }

    fmt.Println("LogisticRegression")
    dirLR.Predict((&self.LR), self.MLP[self.nMLP - 1].Activated, actualLabel)
}