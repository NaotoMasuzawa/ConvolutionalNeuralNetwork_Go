package main

import(
    
    "fmt"
    "math/rand"
    dirCNN  "./ConvolveNeuralNetwork"
    dirLoad "./LoadMnist"
)

func main(){

    var(
        imageSize = []int {28, 28}
        channel = 1
        nKernel = []int {20, 50}
        kernelSizes = [][]int {{5, 5}, {5, 5}}
        poolSizes = [][]int {{2, 2}, {2, 2}}
        MLPsizes = []int {500}
        class = 10
        miniBatchSize = 10
        dropOut = []bool {false}
        dropOutPossibility = []float64 {0.0}
        activationName = "ReLU"
    )

    fmt.Println()
    fmt.Println("Construct the model.")
    fmt.Println("----------------------------------------")

    var cnn dirCNN.ConvolveNeuralNetwork
    dirCNN.Construct((&cnn), 
                      imageSize, 
                      channel, 
                      nKernel, 
                      kernelSizes, 
                      poolSizes, 
                      MLPsizes, 
                      class, 
                      miniBatchSize, 
                      dropOut, 
                      dropOutPossibility, 
                      activationName)

    var(
        trainImage [][][][]float64 = dirLoad.LoadImage("Train")
        trainLabel [][]int = dirLoad.LoadLabel("Train")
        
        testImage [][][][]float64 = dirLoad.LoadImage("Test")
        testLabel [][]int = dirLoad.LoadLabel("Test")
        
        ImageBatch [][][][]float64
        LabelBatch [][]int

        nTrain int = len(trainImage)
        nTest int = len(testImage)
        perm []int
        learningRate float64 = 0.001
        epochs int = 20
        CrossEntropy []float64
        Accuracy []float64
    )

    ImageBatch = make([][][][]float64, miniBatchSize)
    LabelBatch = make([][]int, miniBatchSize)
    for i := 0; i < miniBatchSize; i++{
        ImageBatch[i] = make([][][]float64, channel)
        LabelBatch[i] = make([]int, class)

        for j:= 0; j < channel; j++{
            ImageBatch[i][j] = make([][]float64, imageSize[0])

            for k := 0; k < imageSize[0]; k++{
                ImageBatch[i][j][k] = make([]float64, imageSize[1])
            }
        }
    }

    CrossEntropy = make([]float64, epochs)
    Accuracy = make([]float64, epochs)
    
    for i := 0; i < epochs; i++{
        perm = rand.Perm(nTrain)

        for count := 0; count < (nTrain / miniBatchSize); count++{
            fmt.Println()
            fmt.Printf("Train Epoch %d Batch %d\n", i + 1, count + 1)
            
            for j := 0; j < miniBatchSize; j++{
                ImageBatch[j] = trainImage[perm[count * miniBatchSize + j]]
                LabelBatch[j] = trainLabel[perm[count * miniBatchSize + j]]
            }
            dirCNN.Train((&cnn), ImageBatch, LabelBatch, learningRate)
            CrossEntropy[i] += cnn.LR.CrossEntropy / float64(nTrain)
        }
        learningRate *= 0.99

        for count := 0; count < (nTest / miniBatchSize); count++{
            fmt.Println()
            fmt.Printf("Test Epoch %d Batch %d\n", i + 1, count + 1)
            
            for j := 0; j < miniBatchSize; j++{
                ImageBatch[j] = testImage[count * miniBatchSize + j]
                LabelBatch[j] = testLabel[count * miniBatchSize + j]
            }
            dirCNN.Test((&cnn), ImageBatch, LabelBatch)
            Accuracy[i] += float64(cnn.LR.AccuracyCount) / float64(nTest)
        }
        
        fmt.Println()
        fmt.Printf("Epoch %d finished.\n", i + 1)
        fmt.Printf("CrossEntropy %f\n", CrossEntropy[i])
        fmt.Printf("Accuracy %f\n", Accuracy[i])
    }

    fmt.Println()
    fmt.Println("Results")
    for i := 0; i < epochs; i++{
        fmt.Printf("Epoch %d\n", i + 1)
        fmt.Println("CrossEntropy")
        fmt.Println(CrossEntropy[i])
        fmt.Println("Accuracy")
        fmt.Println(Accuracy[i])
    }
}