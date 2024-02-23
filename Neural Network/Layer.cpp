#include<iostream>
#include<unistd.h>
#include<stdlib.h>
#include<sys/ipc.h>
#include<sys/wait.h>
#include<sys/shm.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<time.h>
#include<string>
#include<cstring>
#include<fcntl.h>
#include<cmath>
#include<pthread.h>
#include<sstream>
#include<fstream>


using namespace std;

pthread_mutex_t m_weights;	
pthread_mutex_t m_prevArr;	

struct Info{
    int weightsCol;
    int weightsRow;
    int InputNum;

    Info(){
        weightsCol =0;
        weightsRow=0;
        InputNum =0;
    }
};


struct PassingStruct{
    Info info;
    double ** weights;
    double * prevAttr;
    int index;

    PassingStruct(){
        index =0;
    }

    PassingStruct(Info & first, double ** & second, double * &third,int index){
        info = first;
        weights = second;
        prevAttr = third;
        this->index = index;
    }

};


void * Neuron(void * args){

    PassingStruct temp = *((PassingStruct*)args);
    
    
    double *Attribute=new double;

    for(int i =0;i<temp.info.InputNum;i++){
        *(Attribute)+=temp.weights[temp.index][i] * temp.prevAttr[i];
    }


    pthread_exit((void*)Attribute);
}


Info IPCSeperate(string& passed,double * & prevAttr, double ** & weights){

    int countComma=0; // num of weights (col)
    int countPerc=0; // num of weights (row);
    int countHash=0; // previous neuron attributes
    
    pthread_mutex_lock(&m_weights);

    for(int i =0;i<passed.length();i++){
        if(passed[i]==',')
            countComma++;
        if(passed[i]=='#')
            countHash++;
        if(passed[i]=='%')
            countPerc++;
    }

    countComma = countComma/countPerc;


    prevAttr = new double[countComma];

    weights = new double *[countPerc];
    for(int i =0;i< countPerc;i++){
        weights[i] = new double[countComma];
    }

	pthread_mutex_unlock(&m_weights);

    string str;
    int rowCount=0;
    int colCount =0;
    int index =0;

    for(int i =0;i<passed.length();i++){

        if(passed[i]!=','&& passed[i]!= '#' && passed[i]!='%')
            str+=passed[i];
        if(passed[i]==','){
            weights[rowCount][colCount]=stof(str);
            str="";
            colCount++;
            if(colCount>=countComma){
                rowCount++;
                colCount = 0;
            }
        }
        if(passed[i]=='#'){
            prevAttr[index] = stof(str);
            str="";
            index++;
        }
    }
    
    Info retVal;

    retVal.weightsCol = countComma;
    retVal.weightsRow = countPerc;
    retVal.InputNum = countHash;

    return retVal;
}

string MakeStr(void ** calculated, int size){
    string str;
    for(int i =0;i<size;i++){
        str+=to_string(*((double*)calculated[i])) + ",";
    }
    return str;
}



int main(int argc,char* argv[]){




    int bufferLength = stoi(argv[1]) + 1; // value passed from execl process
    char * buffer;
    buffer = new char[bufferLength +1];


    char recieveNum = argv[2][0];
    char recieveStat = argv[2][1];

    int np_read;
    np_read = open("PIPE",O_RDONLY);
    read(np_read,buffer,bufferLength);
    close(np_read);
    string str = buffer;
    delete[] buffer;


    if(recieveNum == 'A' && recieveStat == 'B'){
        cout << "Returning " << str << "  Back to Input Layer from Output Layer" <<endl;
        exit(0);
    }
    if(recieveStat == 'B'){
        cout << "Returning " << str << "  Back to Input Layer from Hidden Layer#" << recieveNum << endl;
        exit(0);
    }



    double * prevAttr;
    double ** weights;

    Info info;

    info = IPCSeperate(str,prevAttr,weights);
    
    PassingStruct * pass = new PassingStruct[info.weightsRow];


    pthread_t * Tid;
    Tid = new pthread_t[info.weightsRow];

    cout << "Weights Recieved from Pipe : " << endl;
    for(int i =0;i<info.weightsRow;i++){
        for(int c=0;c<info.weightsCol;c++){
            cout << weights[i][c] << " ";
        }
        cout << endl;
    }

    cout << "Attributes Recieved from Pipe : " << endl;
    for(int i=0;i<info.InputNum;i++){
        cout << prevAttr[i] << " ";
    }
    cout << endl <<endl <<endl;

    for(int i =0;i<info.weightsRow;i++){ // weights row tell the number of neurons
        pass[i].info = info;
        pass[i].weights = weights;
        pass[i].prevAttr = prevAttr;
        pass[i].index = i;
        pthread_create(&Tid[i],NULL,Neuron,(void*)&pass[i]);
    }
    
    void ** retVals = new void * [info.weightsRow];
    for(int i =0;i<info.weightsRow;i++){
        pthread_join(Tid[i],&retVals[i]);

    }

    
        str = MakeStr(retVals,info.weightsRow);
        buffer = new char[500];
        strcpy(buffer,str.c_str());
    
        int np_write2 = open("PIPE2",O_WRONLY);
        write(np_write2,buffer,500);
        close(np_write2);
        string pipe_send;
        char* PipeStr;

        for(int i =0;i<info.weightsRow;i++)
            delete[] weights[i];
        delete[] weights;
        delete[] prevAttr;
        delete[] Tid;
        delete[] retVals;
        delete[] buffer;


    exit(0);

    return 0;
}
