#include <opencv2/core.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "results_writer.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// ############## Parametri globali ###############

bool debug = true;

bool use_mask = false;
Mat result;

Mat resTM, resFM;

const char* image_window = "Source Image";
const char* result_window = "Result window";
const char* templName = "Best Model View";
const char* maskName = "Mask View";

static const char* params =
"{ help            | false | print usage                           }"
"{ imagesPath       |       | Images directory                   }";

// ################################################

///////////////////////////////////////////////////

class MatchInfo {
  public:
    Point coords;
    double score;
    int templNumber;
    Mat matchedImage;
    void setMatchedImage(Mat);
    void setCoords(Point);
    void setScore(double);
    Mat getMatchedImage();
    Point getCoords();
    double getScore();
};

bool sortByScore(MatchInfo &A, MatchInfo &B)
{
    return (A.score > B.score);
}

void MatchInfo::setMatchedImage(Mat result){
  this->matchedImage = result;
  return;
}

void MatchInfo::setCoords(Point coordinate){
  this->coords = coordinate;
  return;
}

void MatchInfo::setScore(double punteggio){
  this->score = punteggio;
  return;
}

Mat MatchInfo::getMatchedImage(){
  return this->matchedImage;
}

Point MatchInfo::getCoords(){
  return this->coords;
}

double MatchInfo::getScore(){
  return this->score;
}

////////////////////////////////////////////////////

double median( Mat channel )
{
    double m = (channel.rows*channel.cols) / 2;
    int bin = 0;
    double med = -1.0;

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    
    cv::calcHist( &channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

    for ( int i = 0; i < histSize && med < 0.0; ++i )
    {
        bin += cvRound( hist.at< float >( i ) );
        if ( bin > m && med < 0.0 )
            med = i;
    }

    return med;
}

Mat doCannyTemplate(Mat src, int kernel_size){

  Mat edge_map, src_gray;

  // Conversione in grayscale
  cvtColor( src, src_gray, COLOR_RGB2GRAY );

  Canny(src_gray, edge_map, 0, 100, kernel_size);

  return edge_map;
}

Mat doCannyImage(Mat src, int kernel_size){

  Mat src_gray, edge_map, dst, _img;

  // Conversione in grayscale
  cvtColor( src, src_gray, COLOR_RGB2GRAY );
  blur(src_gray, dst, Size(3, 3));

  long double T_otsu = cv::threshold(dst, _img, 0, 255, THRESH_BINARY | THRESH_OTSU);
  //if(debug){
  //  cout << "Otsu Threshold : " << T_otsu << endl;
  //}

  Canny(dst, edge_map, T_otsu*0.6 , T_otsu*0.8 , kernel_size);

  return edge_map;
}

Mat MatchingMethod( Mat img, Mat templ, Mat mask, int match_method, MatchInfo* score )
{
  Mat img_display, template_gray, img_gray;
  double minVal, maxVal; 
  Point minLoc, maxLoc, matchLoc;
  int result_cols, result_rows;

  img.copyTo( img_display );

  result_cols =  img.cols - templ.cols + 1;
  result_rows = img.rows - templ.rows + 1;

  result.create( result_rows, result_cols, CV_32FC1 );

  // Calcolo edge map tramite Canny
  template_gray = doCannyTemplate(templ, 3);
  img_gray = doCannyImage(img, 3);

  /*if(debug)
  {
    imshow("TEMPL EM", template_gray);
    imshow("IMAGE EM", img_gray);

    waitKey(0);
  }*/

  // Chiamo il metodo di template matching
  matchTemplate( img_gray, template_gray, result, match_method);

  // Recupero valori dei punti di max e min
  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  
  // Salvo punto massimo voluto da
  matchLoc = maxLoc;

  // Salvo il punteggio e relativo punto
  score->setScore(maxVal);
  score->setCoords(matchLoc);
  
  // Crop del risultato
  Mat subImage(img_display, cv::Rect(matchLoc.x, matchLoc.y, matchLoc.x + templ.cols - matchLoc.x, matchLoc.y + templ.rows - matchLoc.y));

  // Salvo risultato
  img_display.copyTo( resTM );

  return subImage;
}

vector<String> loadImages(String imagesPath, vector<Mat>* images)
{

  vector<String> imagesName;

  try
  {
    vector<String> fn;
    
    glob(imagesPath, fn);
    size_t count = fn.size();

    //load the images in the dir
    for (size_t i = 0; i < count; i++) {
      
      string token(fn[i].substr(fn[i].rfind("/") + 1));
      imagesName.push_back(token);

      Mat tmp = imread(fn[i], IMREAD_COLOR);
    if (!tmp.empty())
      images->push_back(tmp);
    }

  }
  catch (const Exception& e)
  {
    cerr << "Error during the loading of the images. Reason: " << e.msg << endl;
    exit(1);
  }

  return imagesName;
}

void generateLogFile(vector<String> models_name, vector<String> tests_name, vector<vector<MatchInfo>> M, String object)
{
  ResultsWriter res_writer(object);
  
  for(int i = 0; i < 10; i++)
  {
    //cout << "DAVIDE-DEBUG: scrivo risultati per immagine: " << tests_name[i] << endl;

    for(int j = 0; j < 10; j++)
    {
      //cout << "DAVIDE-DEBUG: aggiungo il risultato [" << j << "] - " << models_name[M[i][j].templNumber] << endl;
      if( !res_writer.addResults( tests_name[i], models_name[M[i][j].templNumber], M[i][j].coords.x, M[i][j].coords.y ) ){
        cout << "[ERROR] - Problem during adding result for: test image = " << tests_name[i] << " template = " << models_name[M[i][j].templNumber] << std::endl;
        return;
      }
    }
  }

  //cout << "DAVIDE-DEBUG: scrivo risultati per " << object << endl;
  if( !res_writer.write() ){
    cout << "[ERROR] - Problem during write of the result file for " << object << std::endl;
    return;
  }
}

double doSSIM(Mat img, Mat temp) {

  double C1 = 6.5025, C2 = 58.5225;
  double mx = 0, my = 0, varX = 0, varY = 0, varXY = 0;

  Mat greyImage, greyTempl;

  cvtColor( img, greyImage, COLOR_RGB2GRAY );
  cvtColor( temp, greyTempl, COLOR_RGB2GRAY );

  int width = greyImage.cols;
  int height = greyImage.rows;
  
  for (int v = 0; v < height; v++)
  {
    for (int u = 0; u < width; u++)
    {
      mx += greyImage.at<uchar>(v, u);
      my += greyTempl.at<uchar>(v, u);

    }
  }

  mx = mx / width / height;
  my = my / width / height;

  for (int v = 0; v < height; v++)
  {
    for (int u = 0; u < width; u++)
    {
      varX += (greyImage.at<uchar>(v, u) - mx)* (greyImage.at<uchar>(v, u) - mx);
      varY += (greyTempl.at<uchar>(v, u) - my)* (greyTempl.at<uchar>(v, u) - my);
      varXY += abs((greyImage.at<uchar>(v, u) - mx)* (greyTempl.at<uchar>(v, u) - my));
    }
  }
  
  varX = varX / (width*height - 1);
  varY = varY / (width*height - 1);
  varXY = varXY / (width*height - 1);

  double zi = (2 * mx*my + C1) * (2 * varXY + C2); 
  double mu = (mx*mx + my * my + C1) * (varX + varY + C2);
  double ssim = zi / mu;
  
  return ssim;
}

double compareImages(Mat image, Mat tmp, Mat mask){

  double similarity = 0;

  //Applico maschera del template per selezionare solo porzione che mi serve
  Mat filtered = cv::Mat::zeros(image.size(), image.type());    
  image.copyTo(filtered, mask);
  
  similarity = doSSIM(filtered, tmp);
  //cout << "DAVIDE-DEBUG: res1 = " << res1 << " res2 = " << res2 << endl;

  return similarity;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main( int argc, char** argv )
{
  Mat img, templ, mask, sub_image;
  vector<vector<MatchInfo>> score_matrix;
  MatchInfo scoreInfo;
  String basepath;
  vector<String> modelsName;
  vector<String> testImgsName;
  vector<String> objects = {"can"};

  cv::CommandLineParser parser(argc, argv, params);
  if (parser.get<bool>("help"))
  {
    parser.printMessage();
    return 0;
  }
  
  if (!parser.has("imagesPath"))
  {
    cout << "FAILED TO READ MANDATORY CMD LINE ARGS!" << endl;
    cout << " you should specify the imagePath cmd line args!";

    waitKey(0);
    return 1;
  }
  
  String mainDir = parser.get<cv::String>("imagesPath");
  
  cout << "Input path: " << mainDir << endl;


  for(size_t o = 0; o < objects.size(); o++){

    basepath = mainDir + "/" + objects[o];
    cout << endl << "#### RUN ALGORITHM FOR <" << objects[o] << "> ####" << endl;

     // Load test images
    String imagePath = basepath + "/test_images/*.jpg";
    cout << "#### UPLOADING TEST IMAGES: " << imagePath << " ####" << endl;
    vector<Mat> testImages;
    testImgsName = loadImages(imagePath, &testImages);

    // Load models
    String templatePath = basepath + "/models/model*";
    cout << "#### UPLOADING MODELS: " << templatePath << " ####" << endl;
    vector<Mat> templates;
    modelsName = loadImages(templatePath, &templates);

    // Load masks
    String maskPath = basepath + "/models/mask*";
    cout << "#### UPLOADING MASKS: " << maskPath << " ####" << endl << endl;
    vector<Mat> masks;
    loadImages(maskPath, &masks);

    // Scorro per immagine di test
    for(size_t t = 0; t < testImages.size(); t++)
    {
      
      // Reset delle strutture dati
      std::vector<MatchInfo> temp;
      std::vector<MatchInfo> filteredResults;

      // Scorro per modelli
      for(size_t i = 0; i < templates.size(); i++)
      {
        // Faccio template matching
        scoreInfo.templNumber = i;
        sub_image = MatchingMethod( testImages[t], templates[i], masks[i], TM_CCOEFF, &scoreInfo);

        // Salvo risultato del template match
        scoreInfo.setMatchedImage(sub_image);
        
        // Salvo score e coordinate del match
        temp.push_back(scoreInfo);

      }
      //score_matrix.push_back(temp);

      // Ordino risultati
      cout << "#### SORTING RESULTS ####" << endl << endl;
      std::sort(temp.begin(), temp.end(), sortByScore);

      // Discrimino falsi positivi
      cout << "#### FILTERING FROM BEST 30 RESULTS ####" << endl << endl;
      for(int k = 0; k < 30; k++)
      {
          MatchInfo m;
          double ssimScore = compareImages(temp[k].getMatchedImage(), templates[temp[k].templNumber], masks[temp[k].templNumber]);

          m.setMatchedImage(temp[k].getMatchedImage());
          m.setCoords(temp[k].getCoords());
          m.setScore(ssimScore);
          m.templNumber = temp[k].templNumber;

          filteredResults.push_back(m);
      }
      //Ordino nuovo vettore di risultati
      std::sort(filteredResults.begin(), filteredResults.end(), sortByScore);

      cout << "#### SHOW BEST 10 RESULTS FOR "<< testImgsName[t] <<" ####" << endl << endl;

      // Inserisco bounding box per i mogliori risultati
      for(int j = 0; j < 10; j++)
      {
          cout << "MATCH [" << j+1 <<"]: " << modelsName[filteredResults[j].templNumber] << " SCORE = " << filteredResults[j].score << " POINT = (" << filteredResults[j].coords.x << ", " << filteredResults[j].coords.y << ")" << endl;

          // Disegno rettangolo
          rectangle( testImages[t], filteredResults[j].coords, Point( filteredResults[j].coords.x + templates[filteredResults[j].templNumber].cols , filteredResults[j].coords.y + templates[filteredResults[j].templNumber].rows ), Scalar(0 + 10*j, 255, 0), 2, 8, 0 );
          putText( testImages[t], to_string(j+1), filteredResults[j].coords, FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(100,255,0), 2.0 );
      }

      //Salvo risultato
      score_matrix.push_back(filteredResults);

      // Mostra risultati solo in DEBUG MODE
      if(debug)
      {
        namedWindow( image_window, WINDOW_AUTOSIZE );
        namedWindow( templName, WINDOW_AUTOSIZE );

        imshow(image_window, testImages[t]);
        imshow(templName, templates[filteredResults[0].templNumber]);

        waitKey(0);
      }
    }

    // Generating log file
    cout << endl << "#### GENERATING LOG FILE " << objects[o] << "_result.txt" << " ####" << endl;
    generateLogFile(modelsName, testImgsName, score_matrix, objects[o]);
  }

  sleep(1);
  
  return 0;
}