/*
I used the following github code.
https://github.com/petar/GoMNIST
I am grateful to Petar Maymounkov.
*/

package LoadMnist

import(
    
    "compress/gzip"
    "encoding/binary"
    "fmt"
    "io"
    "os"
)

const(
    
    trainImageFile = "LoadMnist/train-images-idx3-ubyte.gz"
    testImageFile  = "LoadMnist/t10k-images-idx3-ubyte.gz"
    trainLabelFile = "LoadMnist/train-labels-idx1-ubyte.gz"
    testLabelFile  = "LoadMnist/t10k-labels-idx1-ubyte.gz"
    
    imageMagic = 0x00000803
    labelMagic = 0x00000801
    Width      = 28
    Height     = 28
)

type(
    RawImage []byte
    Label uint8
)

func ReadImageFile(name string) (rows, cols int, imgs []RawImage, err error) {
    
    f, err := os.Open(name)
    if err != nil {
        fmt.Println("ERROR")
        return 0, 0, nil, err
    }
    defer f.Close()
    z, err := gzip.NewReader(f)
    if err != nil {
        return 0, 0, nil, err
    }
    return readImageFile(z)
}

func readImageFile(r io.Reader) (rows, cols int, imgs []RawImage, err error) {
    
    var (
        magic int32
        n     int32
        nrow  int32
        ncol  int32
    )
    if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
        return 0, 0, nil, err
    }
    if magic != imageMagic {
        return 0, 0, nil, os.ErrInvalid
    }
    if err = binary.Read(r, binary.BigEndian, &n); err != nil {
        return 0, 0, nil, err
    }
    if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
        return 0, 0, nil, err
    }
    if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
        return 0, 0, nil, err
    }
    imgs = make([]RawImage, n)
    m := int(nrow * ncol)
    for i := 0; i < int(n); i++ {
        imgs[i] = make(RawImage, m)
        m_, err := io.ReadFull(r, imgs[i])
        if err != nil {
            return 0, 0, nil, err
        }
        if m_ != int(m) {
            return 0, 0, nil, os.ErrInvalid
        }
    }
    return int(nrow), int(ncol), imgs, nil
}

func ReadLabelFile(name string) (labels []Label, err error) {
    
    f, err := os.Open(name)
    if err != nil {
        return nil, err
    }
    defer f.Close()
    z, err := gzip.NewReader(f)
    if err != nil {
        return nil, err
    }
    return readLabelFile(z)
}

func readLabelFile(r io.Reader) (labels []Label, err error) {
    
    var (
        magic int32
        n     int32
    )
    if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
        return nil, err
    }
    if magic != labelMagic {
        return nil, os.ErrInvalid
    }
    if err = binary.Read(r, binary.BigEndian, &n); err != nil {
        return nil, err
    }
    labels = make([]Label, n)
    for i := 0; i < int(n); i++ {
        var l Label
        if err := binary.Read(r, binary.BigEndian, &l); err != nil {
            return nil, err
        }
        labels[i] = l
    }
    return labels, nil
}

func LoadImage(name string)(Image [][][][]float64){

    var LoadName string
    if name == "Train"{
        LoadName = trainImageFile
    }else if name == "Test"{
        LoadName = testImageFile
    }

    row, col, im, err := ReadImageFile(LoadName)
    if err != nil{
        fmt.Println("ERROR")
    }

    Image = make([][][][]float64, len(im))
    for i := 0; i < len(im); i++{
        Image[i] = make([][][]float64, 1)

        Image[i][0] = make([][]float64, col)
        for j := 0; j < col; j++{
            Image[i][0][j] = make([]float64, row)
        }
    }

    for i := 0; i < len(im); i++{
        index := 0
        for j := 0; j < col; j++{
            for k := 0; k < row; k++{
                Image[i][0][j][k] = float64(im[i][index]) / 255
                index += 1
            }
        }
    }
    return Image
}

func LoadLabel(name string)(Label [][]int){

    var LoadName string
    if name ==  "Train"{
        LoadName = trainLabelFile
    }else if name == "Test"{
        LoadName = testLabelFile
    }

    label, err := ReadLabelFile(LoadName)
    if err != nil{
        fmt.Printf("Load %s label ERROR\n", name)
    }

    Label = make([][]int, len(label))
    for i := 0; i < len(label); i++{
        Label[i] = make([]int, 10)
    }

    for i := 0; i < len(label); i++{
        Label[i][int(label[i])] = 1
    }

    return Label
}