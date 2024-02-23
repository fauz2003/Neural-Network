#include<iostream>
#include<unistd.h>
#include<stdlib.h>
#include<sys/ipc.h>
#include<sys/wait.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<sys/shm.h>
#include<time.h>
#include<cstring>
#include<string>
#include<fcntl.h>
#include<pthread.h>
#include<fstream>
#include<cerrno>

using namespace std;

/*
    Points to remember:
    1) each layer is a seperate process
    2) each neuron is a seperate thread
    3) IPC through pipes
    4) Pass weights and bias through pipes
    5) Weight Matrix and other bias can be in shared memory
*/

/*
    The Hidden layer process will get passed all the attribute values
    of the previous layer neurons, and the weights that are connected
    to it through pipes.
*/

pthread_mutex_t m_init;				//mutex for inpToHid[]
pthread_mutex_t m_HtoO;				//mutex for HToO[]
pthread_mutex_t m_Hid;				//mutex for HidMatrix[]
pthread_mutex_t m_valPass;			//mutex for valPass
pthread_mutex_t m_HlayerAttr;		//mutex for HLayerAttr[]




void InitialValues(double ** InpToHid,double ** HtoO,double*** HidMatrix,double* &inputAttr,int InputNeurons,int NeuPerL,int OutputNeurons,int numL){

    string str;
    string tempstr;
    int colCount =0,rowCount=0,pageCount =0;
    ifstream fin;
    fin.open("Initial.txt");
    getline(fin,str,'\n');
    
    
	 pthread_mutex_lock(&m_init);
    for(int c=0;c<NeuPerL;c++){
        getline(fin,str,'\n');
        for(int i =0;i<str.length();i++){
            if(str[i]!=','){
                tempstr+= str[i];
            }
            else{
                InpToHid[rowCount][colCount] = stof(tempstr);
                tempstr="";
                colCount++;
                if(colCount == InputNeurons){
                    colCount = 0;
                    rowCount++;
                }
            }
        }
    }
    pthread_mutex_unlock(&m_init);

	 pthread_mutex_lock(&m_Hid);
    for(int k=0;k<(numL-1);k++){
        colCount =0;
        rowCount=0;
        getline(fin,str,'\n'); // useless line in txt file
        getline(fin,str,'\n'); // useless line in txt file
        tempstr="";
        for(int j=0;j<NeuPerL;j++){
            getline(fin,str,'\n');
            for(int i =0;i<str.length();i++){
                if(str[i]!=','){
                    tempstr+=str[i];
                }
                else{
                    HidMatrix[pageCount][rowCount][colCount]= stof(tempstr);
                    colCount++;
                    tempstr="";
                    if(colCount==NeuPerL){
                        colCount =0;
                        rowCount++;
                        if(rowCount==NeuPerL){
                            rowCount =0;
                            pageCount++;
                        }
                    }
                }
            }
        }
    }
    pthread_mutex_unlock(&m_Hid);

	 pthread_mutex_lock(&m_HtoO);
    getline(fin,str,'\n');
    getline(fin,str,'\n'); // not needed lines

    colCount =0;
    rowCount =0;
    pageCount =0;
    tempstr ="";
    for(int j =0;j<OutputNeurons;j++){
        getline(fin,str,'\n');
        for(int i =0;i<str.length();i++){
            if(str[i]!=','){
                tempstr += str[i];
            }
            else{
                HtoO[rowCount][colCount] = stof(tempstr);
                colCount++;
                tempstr ="";
                if(colCount==NeuPerL){
                    colCount =0;
                    rowCount++;
                }
            }
        }
    }
    pthread_mutex_unlock(&m_HtoO);

    rowCount=0;
    colCount=0;

    getline(fin,str,'\n');
    getline(fin,str,'\n');
    getline(fin,str,'\0'); // last line
    tempstr="";
    for(int i =0;i<str.length();i++){
        if(str[i]!=','){
            tempstr+=str[i];
        }
        else{
            inputAttr[colCount] = stof(tempstr);
            colCount++;
            tempstr ="";
        }
    }
    
    fin.close();

    
}


void IPCSignal(string & str,double ** Weights,double * prevAttr, int wRow,int wCol){
    str ="";

	 pthread_mutex_lock(&m_valPass);
    for(int r=0;r<wRow;r++){
        for(int c=0;c<wCol;c++){
            str+= to_string(Weights[r][c]);
            str+=",";
        }
        str+="%";// weight rowend
    }
    for(int i =0;i<wCol;i++){
        str+=to_string(prevAttr[i]);
        str+= "#";
    }
    pthread_mutex_unlock(&m_valPass);
}


void AssignAttr(double * &tempAttr,string passed){
    string str;
    int index=0;
    for(int i =0;i<passed.length();i++){
        if(passed[i]!=',')
            str+= passed[i];
        else{
            tempAttr[index] = stof(str);
            str="";
            index++;
        }
    }
}


void NeuralNetwork(int InputNeurons,int numL, int neuPerL,int OutputNeurons,double * inputs){

    double ** ItoH= new double*[neuPerL]; // Weight Matrix from Input Layer to Hidden Layer
    for(int i =0;i<neuPerL;i++)
        ItoH[i] = new double[InputNeurons];

    double** HtoO = new double * [OutputNeurons]; // Weight Matrix from last HIdden layer to Output
    for(int i =0;i<OutputNeurons;i++)
        HtoO[i] = new double [neuPerL];

    double *** HidMatrix = NULL; //Weights Between Hidden Layers
    if(numL>1){
        double *** temp = new double ** [numL-1];
        for(int i =0;i<(numL-1);i++){
            temp[i] = new double *[neuPerL]; // Weight Matrix within hidden layers
        }
        for(int r=0;r<(numL-1);r++){
            for(int c=0;c<neuPerL;c++){
                temp[r][c] = new double[neuPerL];
            }
        }
        HidMatrix = temp;
    }
    InitialValues(ItoH,HtoO,HidMatrix,inputs,InputNeurons,neuPerL,OutputNeurons,numL); // assigning initial weights as random

    string valPass;

    double * HLayerAttributes= new double[neuPerL];
    double * FinalOutput = new double[OutputNeurons];
    
    
    int valPassLength;
    string tempstr;
    char * pipeBuffer;

    int np_write;
    int rd_open;

    int var1,var2;
    char c_process='0';


    unlink("PIPE");
    mkfifo("PIPE",0666);
    unlink("PIPE2");
    mkfifo("PIPE2",0666);
    unlink("PIPE3");
    mkfifo("PIPE3",0666);

    char * PipeStr;
    char * PNum = new char[3];
    PNum[1] ='F';
    PNum[2] = '\0';
    for(int repeat=0;repeat<2;repeat++){
    if(!fork()){
            PNum[1] = 'F';
        for(int i = 0;i<=numL;i++){
            PNum[0] = i+'0';
            if(i==0){ // send ItoH matrix and input neurons and neuPerL
                cout << "Hidden Layer#" << (i+1) << " " <<endl;
                IPCSignal(valPass,ItoH,inputs,neuPerL,InputNeurons);
            }
            else{
                if(i!=numL){
                    cout << "Hidden Layer#" << (i+1) << " " << endl;
                    IPCSignal(valPass,HidMatrix[i-1],HLayerAttributes,neuPerL,neuPerL);
                }
                else{
                    cout << "Output Layer " << endl;
                    IPCSignal(valPass,HtoO,HLayerAttributes,OutputNeurons,neuPerL);
                    PNum[0]= 'A';
                }
            }

            valPassLength = valPass.length();

            pipeBuffer = new char[valPassLength +1];
            strcpy(pipeBuffer,valPass.c_str()); // passing value

            if(!fork()){  // make Layer Processes
                // exec it
                char strLen[5] = {'\0',};
                    tempstr = to_string(valPass.length());
                    for(int c=0;c<valPassLength;c++){
                        strLen[c] = tempstr[c];
                    }
                strLen[valPassLength]='\0';
                execl("./L","L",strLen,PNum,NULL); // exit(0) inside exec
            }
            else{
                np_write = open("PIPE",O_WRONLY);
                write(np_write,pipeBuffer,(valPassLength+1));
                close(np_write);
                delete[] pipeBuffer;

                pipeBuffer = new char[500];
                rd_open = open("PIPE2",O_RDONLY);
                read(rd_open,pipeBuffer,500);
                close(rd_open);
                wait(NULL);
                valPass = pipeBuffer;
                if(i!=numL){
                    AssignAttr(HLayerAttributes,valPass);
                }
                else{
                    np_write = open("PIPE3",O_WRONLY);
                    write(np_write,pipeBuffer,500); // passing back the final output
                    close(np_write);
                }
            }
        }
        exit(0);
    }
    else{ // back to main, meaning forward propogation is done
        pipeBuffer = new char[500];
        rd_open = open("PIPE3",O_RDONLY);
        read(rd_open,pipeBuffer,500);
        close(rd_open);
        wait(NULL);
        valPass = pipeBuffer;
        AssignAttr(FinalOutput,valPass);
        cout << "The Final Output Layer Attributes = ";
        for(int i =0;i<OutputNeurons;i++){
            cout << FinalOutput[i] << " ";
        }
        cout << endl;

        double x = FinalOutput[0];
        double f1 = (x*x + x +1)/2;
        double f2 = (x*x -x)/2;

        valPass = "";
        valPass +=to_string(f1);
        valPass+=',';
        valPass+=to_string(f2);
        delete[] pipeBuffer;

        char * NumPassed = new char[4];
        NumPassed[0] = '5';NumPassed[1] = '0';NumPassed[2] = '0';NumPassed[3] = '\0';

        if(repeat==0){
            pipeBuffer = new char[500];
            strcpy(pipeBuffer,valPass.c_str());
            pipeBuffer[valPass.length()] = '\0';
            for(int i =numL;i>0;i--){
                PNum[0] =(i) + '0';
                PNum[1] = 'B';
                if(!fork()){
                    execl("./L","L",NumPassed,PNum,NULL);
                }
                else{
                    np_write =open("PIPE",O_WRONLY);
                    write(np_write,pipeBuffer,500);
                    close(np_write);
                    wait(NULL);
                }
            }
            delete[] pipeBuffer;
            inputs[0] = f1;
            inputs[1] = f2;
        }
    

    }


    }
    for(int i = 0;i<neuPerL;i++)
        delete[] ItoH[i];
    delete[] ItoH;

    for(int i =0;i<OutputNeurons;i++)
        delete[] HtoO[i];
    delete [] HtoO;

    if(numL>1){
        for(int r=0;r<(numL-1);r++){
            for(int c=0;c<neuPerL;c++){
                delete[] HidMatrix[r][c];
            }
        }
        for(int i=0;i<(numL-1);i++)
            delete[] HidMatrix[i];
        delete[] HidMatrix;
    }
    
    delete[] HLayerAttributes;
    

}

int main(){

    system("g++ -o L Layer.cpp -lpthread"); // compile the Layer cpp file just incase it wasnt before

    int numL = 5; // Number of Hidden Layers
    int neuPerL = 8; // Number of neurons per hidden layer 
    int InputNeurons = 2; // Number of Neurons in Input Layer
    int OutputNeurons=1; // Number of Nurons in Output Layer


    double * inputVal = new double[InputNeurons];


    NeuralNetwork(InputNeurons,numL,neuPerL,OutputNeurons,inputVal);

    delete[] inputVal;
    
    return 0;
}
