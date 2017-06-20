#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (sizeof(GlobalX1Arr)/sizeof(int))
#define NBINS 6

int GlobalK = 2;
float GlobalHband = 0.4;
float GlobalX1Arr[14] = {3.0,1.0,5.0,0.0,3.0,5.0,1.0,2.0,2.0,3.0,0.0,1.0,4.0,5.0};
float GlobalX2Arr[14] = {0.0,1.0,1.0,2.0,2.0,5.0,3.0,3.0,4.0,4.0,5.0,5.0,5.0,5.0};
int GlobalCount = 0;

float GlobalR1Arr[NBINS];
float GlobalR2Arr[NBINS];
float GlobalPdfArr[NBINS][NBINS] = {0};
float GlobalScaleArr[NBINS][NBINS];

float GlobalHistArr[NBINS][NBINS];
int GlobalNSingle=0;
int GlobalGridaxisArr[2][NBINS*NBINS] = {0};

void getR1R2()
{
	int i;
	float minValX1 = GlobalX1Arr[0];
	float minValX2 = GlobalX2Arr[0];
	float maxValX1 = GlobalX1Arr[0];
	float maxValX2 = GlobalX2Arr[0];
        /* Find the min and max of the data */
	for(i=1;i<N;i++)
	{
		if(GlobalX1Arr[i] < minValX1)
		{
			minValX1 = GlobalX1Arr[i];
		}
		else if(GlobalX1Arr[i] > maxValX1)
		{
			maxValX1 = GlobalX1Arr[i];
		}
		if(GlobalX2Arr[i] < minValX2)
		{
			minValX2 = GlobalX2Arr[i];
		}
		else if(GlobalX2Arr[i] > maxValX2)
		{
			maxValX2 = GlobalX2Arr[i];
		}
	}
        /* Get the step size of bins and create histogram axes with this */
	float r1Step = (maxValX1 - minValX1)/(NBINS - 1);
	float r2Step = (maxValX2 - minValX2)/(NBINS - 1);
	for(i=0;i<NBINS;i++)
	{
		GlobalR1Arr[i] = minValX1 + r1Step*i;
		GlobalR2Arr[i] = minValX2 + r2Step*i;
	}
//      Should have axes that look like this
//      GlobalR1Arr = {0 1 2 3 4 5}
//      GlobalR2Arr = {0 1 2 3 4 5}
}

void makeHist()/* Turn the data into a 2D array */
{
	int i=0;
	int j=0;
	int k=0;
	for (i=0; i<N; i++)/* For each data pair */
	{
		for (j=0; j<NBINS; j++)
		{
			for (k=0; k<NBINS; k++)
			{
				if (GlobalX1Arr[i] == j && GlobalX2Arr[i] == k)/* If the data pair match the hist coordinate*/
				{
					GlobalHistArr[j][k] = GlobalHistArr[j][k]+1;
					GlobalScaleArr[j][k] = GlobalHistArr[j][k]/N;
				}
			}
		}	
	}
//      Should have
//      GlobalHistArr = {{0 0 1 0 0 1},
//                       {0 1 0 1 0 1},
//                       {0 0 0 1 1 0},
//                       {1 0 1 0 1 0},
//                       {0 0 0 0 0 1},
//                       {0 1 0 0 0 2}}
//
//      GlobalScakeArr = {{0.0000 0.0000 0.0714 0.0000 0.0000 0.0714},
//                        {0.0000 0.0714 0.0000 0.0714 0.0000 0.0714},
//                        {0.0000 0.0000 0.0000 0.0714 0.0714 0.0000},
//                        {0.0714 0.0000 0.0714 0.0714 0.0714 0.0000},
//                        {0.0000 0.0000 0.0000 0.0000 0.0000 0.0714},
//                        {0.0000 0.0714 0.0000 0.0000 0.0000 0.1429}}
}

void grid2d()
{
	int i,j=0,k=0;
	for(i=0;i<NBINS*NBINS;i++)
//	This loop should generate an array that looks something like this
//	0 0
//	0 1
//	0 2
//	0 3
//	...
//	0 NBINS-1
//	1 0
//	1 1
//	1 2
//	1 3
//	...
//	...
//	(NBINS-1) (NBINS-1)
	{
		GlobalGridaxisArr[0][i] = j;
		GlobalGridaxisArr[1][i] = k;
		k=k+1;
		if (k==NBINS)
		{
			k=0;
			j=j+1;
			if (j==NBINS)
			{
				j=0;
			}
		}
	}
}

void getNumSingle()
{
	int i,j;
	for (i=0;i<NBINS;i++)//Get number of hist bins with data
	{
		for (j=0;j<NBINS;j++)
		{
			if (GlobalScaleArr[i][j]>0)
			{
			GlobalNSingle=GlobalNSingle+1; /* Should be 13 */
			}
		}
	}

}

void getTemp(int * tempPtr, float * histgt0Ptr)
{
	int i,j;
	int k=0;
	for (i=0;i<NBINS;i++)
	{
		for (j=0;j<NBINS;j++)
		{
			if (GlobalScaleArr[i][j]>0)
			{
				tempPtr[GlobalCount] = k;
				histgt0Ptr[GlobalCount] = GlobalScaleArr[i][j];
				GlobalCount = GlobalCount+1;
			}
			k=k+1;
		}
	}
//	tempPtr = {2 5 7 9 11 15 16 18 20 22 29 31 35}
//	histgt0Ptr = {0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.0714 0.1429 }
}

void getHistaxis(int *tempPtr, int ** histaxisPtr)
{
	int i,j;
	for (i=0;i<2;i++)
	{
		histaxisPtr[i] = (int*)malloc(sizeof(int)*GlobalCount);
	}
	for (j=0;j<GlobalCount;j++)
	{
		histaxisPtr[0][j] = GlobalGridaxisArr[0][tempPtr[j]];
		histaxisPtr[1][j] = GlobalGridaxisArr[1][tempPtr[j]];
	}
//	histaxisPtr = {	{0 2},
//			{0 5},
//			{1 1},
//			{1 3},
//			{1 5},
//			{2 3},
//			{2 4},
//			{3 0},
//			{3 2},
//			{3 4},
//			{4 5},
//			{5 1},
//			{5 5}}
}


float ascend(float * distancePtr, int len)// Puts the distances from reference point to all data points into ascending order 
{
	int i=0;
	int j=0;
	for (i=0; i<len; i++)
	{
		for (j=0; j<len; j++)
		{
			if (distancePtr[i] < distancePtr[j])
			{

				float t = distancePtr[i];
				distancePtr[i] = distancePtr[j];
				distancePtr[j] = t;
			}
		}
	}
	float kthVal = distancePtr[GlobalK-1];
	return kthVal;
}

void distCalc(int ** histaxisPtr, float * kthDistPtr)// Calculates the distance from each grid point to each data point, calling ascend() to order them 
{
	int i,j;
	float * distancePtr = (float*)malloc(sizeof(int)*GlobalCount);
	for (i=0;i<GlobalCount;i++)
	{
		for (j=0;j<GlobalCount;j++)
		{
			distancePtr[j] = sqrt(pow(histaxisPtr[0][i]-histaxisPtr[0][j],2)+pow(histaxisPtr[1][i]-histaxisPtr[1][j],2));
		}
		kthDistPtr[i] = ascend(distancePtr, GlobalCount);
	}	
	free(distancePtr);
}

void getHwidth(float * kthDistPtr, float GlobalHband, float * hwidthPtr)
{
	int i;
	for(i=0;i<GlobalCount;i++)
	{
		hwidthPtr[i] = GlobalHband * kthDistPtr[i];
	}
//	For K=2
//	kthDistPtr = {1.4142 1.0000 1.4142 1.0000 1.0000 1.0000 1.0000 2.0000 1.4142 1.0000 1.0000 2.2361 1.0000}
}

void getPDF(float * hwidthPtr, float * histgt0Ptr, int ** histaxisPtr)
{
	int i,j,k;
	float gauArr[GlobalCount];
	double exponArr[GlobalCount];
	for(i=0;i<NBINS;i++)
	{
		for(j=0;j<NBINS;j++)
		{
			float sumGau = 0;
			for(k=0;k<GlobalCount;k++)
			{
				exponArr[k] = exp((double)(pow((float)GlobalR1Arr[i] - (float)histaxisPtr[0][k],2) + pow((float)GlobalR2Arr[j] - (float)histaxisPtr[1][k],2))*(double)(-0.5/pow(hwidthPtr[k],2)));
				gauArr[k] = histgt0Ptr[k]/(2*M_PI*pow(hwidthPtr[k],2))*exponArr[k];
			}
			for(k=0;k<GlobalCount;k++)
			{
				sumGau = sumGau + gauArr[k];
			}
			GlobalPdfArr[i][j] = sumGau;
			printf("%f ",GlobalPdfArr[i][j]);
		}
		printf("\n");
	}
//	The final PDF should be (or its transpose):
//	GlobalPdfArr = {{0.0016 0.0149 0.0372 0.0106 0.0035 0.0742},
//			{0.0082 0.0375 0.0183 0.0760 0.0097 0.0743},
//			{0.0097 0.0128 0.0127 0.0790 0.0776 0.0064},
//			{0.0185 0.0168 0.0371 0.0139 0.0745 0.0064},
//			{0.0122 0.0129 0.0119 0.0023 0.0066 0.0774},
//			{0.0084 0.0146 0.0077 0.0012 0.0064 0.1452}}
}


int main()
{	
	freopen("input.txt","r",stdin);

	getR1R2();

	/* Turn data into 2D array */
	makeHist();
	grid2d();
	getNumSingle();

	float * histgt0Ptr = (float*)malloc(sizeof(float)*GlobalNSingle);
	int * tempPtr = (int*)malloc(sizeof(int)*GlobalNSingle);
	getTemp(tempPtr, histgt0Ptr);

	int ** histaxisPtr = (int**)malloc(sizeof(int*)*2);
	getHistaxis(tempPtr,histaxisPtr);

	free(tempPtr);

	float * kthDistPtr = (float*)malloc(sizeof(float)*GlobalCount);
	distCalc(histaxisPtr, kthDistPtr);

	float * hwidthPtr = (float*)malloc(sizeof(float)*GlobalCount);
	getHwidth(kthDistPtr, GlobalHband, hwidthPtr);

	free(kthDistPtr);

	getPDF(hwidthPtr, histgt0Ptr, histaxisPtr);

	int i;
	for(i=0;i<2;i++)
	{
		free(histaxisPtr[i]);
	}
	free(histaxisPtr);
	free(hwidthPtr);
	free(histgt0Ptr);
	return(0);
}
