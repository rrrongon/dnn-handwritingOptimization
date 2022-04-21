
#  Deep Neural Network for Hand Written Digit Recognition

Outcome: Experience with a common workload on today’s high performance computing (HPC) 
systems - training deep neural networks.

Input: digits of 28*28 image matrix of each digit with label.

Training Output: On each iteration training output provides weights of each possible digit.
Higher the value, more likely to be closer to prediction of that value.

Testing output: Accuracy of correct prediction on each digit.

Dataset: MNIST dataset. Which can found here https://deepai.org/dataset/mnist 

Compile and Run:
```
cd Project_2
make clean&& make
./prject2.x N 
```
Here N is training iteration. Default is 70K. \
Upon a successfull compilation and run, program will start reading training data from *dataset* folder. \
Output.txt: Each training iteration output layer value. \
Outputsample:
```
[Train] Iter 10: err =2.16262, Y = 3
OO[0] = -0.487826
OO[1] = -0.0682834
OO[2] = -0.10499
OO[3] = -0.312949
OO[4] = -0.274636
OO[5] = -1.22069
OO[6] = -0.485087
OO[7] = -0.646399
OO[8] = -0.822935
OO[9] = 0.500752

```
It will start adding output layer results to output.txt. 

**Architecture:** \
Our input image size is 28*28. Thus, Input layer size is 784 neurons. 
First hidden layer N1= 1000. Second hidden layer N2=500. Output layer N3=10. \
Activation Function: A * tanh(B * x). Where A = 1.7159 and B = 0.6666 according to the paper by
Claudiu Ciresan, et al, “Deep Big Simple Neural Nets Excel on Hand-written Digit 
Recognition,” arXiv:1003.0358, Mar, 2010.

**Part 1:** At the very first step is implementing the solution correctly using C++ without any external library.
This means, calculating loops for each neuron to all other neurons on the next layer. Adding bias, Activation
function, and generate next leyer output. \
**Part 2:** Here, improving the previous implementation by Changing memory access pattern, loop interchange, loop unrolling,
Loop blocking. \
**Part 3:** Here, improved the previous implementation by SSE2 optimization on relevant loops those 
provides better computation time comparing to overhead. \
**Part 4:** We have opportunity to improve existing code by added OpenMP thread parallel directives.
We got best result when we ran it on 12 threads, core to core closely binded. \
**Part 5:** Finally, the hardest and most challenging part was to modify the code in such a way
that it can be run on distributed model using MPI. Critical part was domain decomposition. computing
local data for each process and then updating sub-result on the global data that can be accessed by all other 
processes to carry further computations.

**Correctness:** Correctness can be seen by looking into the training output. When the
output of each digit is almost equal to A value mentioned above and only one positive value in the outputs and error rate is very low.
Besides, we can check last few outputs and manually check the prediction. 

**Time and optimization:**  While optimizing it is important to measure which loop is consuming
most of the training executing time and optimize those loops more carefully using mentioned techniques.
To measure time we used *clock_gettime* which can exactly pick the clock cycle for particular thread
from all the jobs running in the machine. For MPI, we used *MPI_time()*. This helped us properly justify where 
to optimize and if any modifications adds more overhead than optimization. Which eventually makes the whole
training time optimal.

**Experimental setup:** We run the program on GenuineIntel Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz
machine.
```
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                56
On-line CPU(s) list:   0-55
Thread(s) per core:    2
Core(s) per socket:    14
Socket(s):             2
NUMA node(s):          2
L1d cache:              32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              35840K
``` 
You can find details of different optimization improvement comparision reports here in the following links. \
[SSE2 SIMD extension](https://docs.google.com/document/d/11OS9S8iDNxVGq_2FlEmJQCHaGvB16eHVKmDA_BBHcM4/edit?usp=sharing) \
[OPENMP](https://docs.google.com/document/d/1X2H3TCE-gfr9iy78gPzDv_9-CQDcStXEh6TsGxozYOQ/edit)


