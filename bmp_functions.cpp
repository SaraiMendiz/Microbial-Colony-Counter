#include "bmp.h"
#include <cmath>
#include <numeric>
#include <algorithm>


void umbral(const cv::Mat& image, const double thresholdValue, cv::Mat& grayscale) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            const cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
            int R = pixel[2];
            int G = pixel[1];
            int B = pixel[0];
            int total = R * R + G * G + B * B;
            if (total < thresholdValue) {
                grayscale.at<unsigned short>(i, j) = 0;
            }
            else {
                grayscale.at<unsigned short>(i, j) = 65535;
            }
        }
    }

}

void etiquetarRegiones(cv::Mat& labeled, std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones) {
    int label = 1; //etiqueta inicial

    for (int y = 0; y < labeled.rows; y++) {
        for (int x = 0; x < labeled.cols; ++x) {
            if (labeled.at<unsigned short>(y, x) == 65535) { //si el pixel es blanco y no est  etiquetado
                std::vector<cv::Point> pointsToLabel; //creamos un vector de puntos para etiquetar los pixeles blancos 
                pointsToLabel.push_back(cv::Point(x, y)); //añadimos el punto a etiquetar

                // Crear un vector de puntos para la nueva etiqueta y añadirlo a regiones.
                regiones.push_back(std::vector<cv::Point>());
                int numberPixels = 0;

                while (!pointsToLabel.empty()) { //mientras haya puntos por etiquetar
                    cv::Point point = pointsToLabel.back(); //obtenemos el  ltimo punto del vector
                    pointsToLabel.pop_back(); //eliminamos el  ltimo punto del vector
                    int x0 = point.x; //obtenemos las coordenadas del punto
                    int y0 = point.y;

                    labeled.at<unsigned short>(y0, x0) = label; //etiquetamos el punto
                    regiones[label - 1].push_back(point); // Ahora el  ndice label-1 deber a existir.

                    for (int i = -1; i <= 1; i++) { //recorremos los puntos vecinos
                        for (int j = -1; j <= 1; j++) {
                            int x1 = x0 + j; //obtenemos las coordenadas del punto vecino
                            int y1 = y0 + i;

                            if (x1 >= 0 && x1 < labeled.cols && y1 >= 0 && y1 < labeled.rows &&
                                labeled.at<unsigned short>(y1, x1) == 65535) {
                                pointsToLabel.push_back(cv::Point(x1, y1));
                            }
                        }
                    }
                    numberPixels++;
                }
                labeledArray.push_back(numberPixels);
                label++;
            }
        }
    }
}

cv::Scalar generarColorAleatorio() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> distrib(0, 255);
    return cv::Scalar(distrib(gen), distrib(gen), distrib(gen)); // BGR
}

void pintarRegiones(cv::Mat& image, const std::vector<std::vector<cv::Point>>& regiones) {
    for (const auto& region : regiones) {
        cv::Scalar color = generarColorAleatorio();
        for (const auto& punto : region) {
            image.at<cv::Vec3b>(punto.y, punto.x) = cv::Vec3b(color[0], color[1], color[2]);
        }
    }
}

// Filtra las regiones que se encuentren dentro del c�rculo m�s grande
void filtrarRegionesDentroDelCirculo(std::vector<std::vector<cv::Point>>& regiones) {
    std::vector<std::vector<cv::Point>> regionesFiltradas; // Vector para almacenar las regiones filtradas

    // Recorre todas las regiones
    for (const auto& region : regiones) {
        std::vector<cv::Point> regionFiltrada; // Vector para almacenar los puntos de la regi�n filtrada
        // Recorre todos los puntos de la regi�n
        for (const auto& punto : region) {
            //si el punto est� dentro del c�rculo, lo a�adimos a la regi�n filtrada, sino lo descartamos
            if (estaDentroDelCirculo(punto)) {
                regionFiltrada.push_back(punto); // A�adir el punto a la regi�n filtrada
            }
        }
        //si la regi�n filtrada no est� vac�a, la a�adimos al vector de regiones filtradas
        if (!regionFiltrada.empty()) {
            regionesFiltradas.push_back(regionFiltrada); // A�adir la regi�n filtrada al vector de regiones filtradas
        }
    }
    // Sustituir el vector de regiones original por el vector de regiones filtradas
    regiones = regionesFiltradas;
}

cv::Point centro;
int radioMaximo = 0;
// Comprueba si un punto est� dentro del c�rculo m�s grande
bool estaDentroDelCirculo(const cv::Point& punto) {
    // Comprueba si la distancia entre el punto y el centro del c�rculo es menor o igual que el radio m�ximo
    //si es menor o igual, el punto est� dentro del c�rculo, sino est� fuera
    return cv::norm(punto - centro) <= radioMaximo;
}

//Emplearemos la transformada de Hough para detectar el c rculo m s grande y as  tomar en cuenta s lo las regiones que se encuentren dentro del mismo

void detectarCirculoMasGrande(cv::Mat& imagen) {
    std::vector<cv::Vec3f> circulos; // Vector de c rculos detectados
    cv::Mat imagenGris; // Imagen en escala de grises
    cv::cvtColor(imagen, imagenGris, cv::COLOR_BGR2GRAY); // Convertir a escala de grises
    cv::GaussianBlur(imagenGris, imagenGris, cv::Size(9, 9), 2, 2); // Aplicar filtro Gaussiano

    // Ajusta estos par metros seg n tus necesidades
    float dp = 1;
    float minDist = imagenGris.rows / 8; // Distancia m nima entre los centros de los c rculos
    float param1 = 100; // Umbral para el detector de bordes interno
    float param2 = 50; // Umbral para el centro del c rculo
    int minRadio = 0; // Radio m nimo a detectar
    int maxRadio = 0; // Radio m ximo a detectar

    cv::HoughCircles(imagenGris, circulos, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadio, maxRadio); // Aplicar transformada de Hough circular

    // Buscar el c rculo m s grande y guardar su centro y radio 
    for (size_t i = 0; i < circulos.size(); i++) {
        cv::Point centroActual(circulos[i][0], circulos[i][1]); // Centro del c rculo actual
        int radioActual = circulos[i][2]; // Radio del c rculo actual

        // Si el radio del c rculo actual es mayor que el radio m ximo encontrado hasta ahora, actualizar el centro y el radio m ximo
        if (radioActual > radioMaximo) {
            centro = centroActual;
            radioMaximo = radioActual;
        }
    }
}


void descartarRegiones(cv::Mat& labeledImage, const float k, std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones) {
    float avg = average(labeledArray);
    float stdDev = standardDeviation(variance(labeledArray, avg));

    // Convertimos los l mites a enteros, realizando solo estas dos conversiones
    int upperLimit = static_cast<int>(std::ceil(avg + k * stdDev));
    int lowerLimit = static_cast<int>(std::floor(avg - k * stdDev));

    // Vector para almacenar las regiones v lidas.
    std::vector<std::vector<cv::Point>> newRegiones;
    std::vector<int> newLabeledArray;

    for (int w = 0; w < labeledArray.size(); w++) {
        // Ahora las comparaciones se realizan entre enteros
        if (labeledArray[w] > 200 && labeledArray[w] <= upperLimit && labeledArray[w] >= lowerLimit) {
            // Conservar la regi n y la etiqueta si es v lida
            newRegiones.push_back(regiones[w]);
            newLabeledArray.push_back(labeledArray[w]);
        }
    }
    // Sustituir los vectores originales por los nuevos vectores
    labeledArray = newLabeledArray;
    regiones = newRegiones;
}

void descartarRegiones2(cv::Mat& labeledImage, std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones) {
    float avg = average(labeledArray);
    float stdDev = standardDeviation(variance(labeledArray, avg));

    // Determinar el porcentaje de regiones que queremos mantener.
    const float targetPercentage = 0.90f;
    int targetRegionCount = static_cast<int>(targetPercentage * labeledArray.size());

    // Probar varios valores de k hasta encontrar el que mantiene el porcentaje deseado.
    float k = 0.0f;
    int keptRegionsCount = 0;
    std::vector<int> newLabeledArray;
    std::vector<std::vector<cv::Point>> newRegiones;

    while (keptRegionsCount < targetRegionCount && k < 10.0f) {
        newLabeledArray.clear();
        newRegiones.clear();

        int upperLimit = static_cast<int>(std::ceil(avg + k * stdDev));
        int lowerLimit = static_cast<int>(std::floor(avg - k * stdDev));

        for (int w = 0; w < labeledArray.size(); w++) {
            if (labeledArray[w] > 200 && labeledArray[w] <= upperLimit && labeledArray[w] >= lowerLimit) {
                newRegiones.push_back(regiones[w]);
                newLabeledArray.push_back(labeledArray[w]);
            }
        }

        keptRegionsCount = newLabeledArray.size();
        k += 0.1f; // Incrementar k en peque�os pasos para encontrar el valor �ptimo.
    }

    // Sustituir los vectores originales por los nuevos vectores.
    labeledArray = newLabeledArray;
    regiones = newRegiones;
}
float average(std::vector<int>& labeledArray) {
    int sum = 0;

    for (int i = 0; i < labeledArray.size(); i++) {
        sum += labeledArray[i];
    }

    return (float)sum / labeledArray.size();
}

float variance(std::vector<int>& labeledArray, float average) {
    int sum = 0;

    for (int i = 0; i < labeledArray.size(); i++) {
        sum += pow(labeledArray[i] - average, 2);
    }

    return sum / labeledArray.size();
}

float standardDeviation(float variance) {
    return sqrt(variance);
}

float average2(std::vector<int>& sizes) {
    int sum = 0;
    for (int size : sizes) {
        sum += size;
    }
    return sizes.empty() ? 0 : static_cast<float>(sum) / sizes.size();
}


void dividirRegionSiEsGrande(std::vector<std::vector<cv::Point>>& regiones, std::vector<int>& sizes, cv::Mat& labeled, float sizeFactor) {
    float avgSize = average(sizes);
    std::vector<std::vector<cv::Point>> newRegiones;
    std::vector<int> newSizes;
    int newLabel = 2;  // Inicia desde 2 para evitar confusión con el fondo

    for (size_t i = 0; i < regiones.size(); i++) {
        auto& region = regiones[i];
        int size = sizes[i];

        if (size > sizeFactor * avgSize && region.size() > 1) {
            int num_parts = std::max(1, (int)std::ceil((float)size / avgSize));
            size_t partSize = region.size() / num_parts;
            size_t remainder = region.size() % num_parts;

            std::sort(region.begin(), region.end(), [](const cv::Point& a, const cv::Point& b) {
                return a.x < b.x; });

            size_t start = 0, end = 0;
            for (int j = 0; j < num_parts; j++) {
                start = end;
                end = start + partSize + (j < remainder ? 1 : 0);
                if (start < region.size() && end <= region.size()) {
                    std::vector<cv::Point> part(region.begin() + start, region.begin() + end);
                    newRegiones.push_back(part);
                    newSizes.push_back(part.size());

                    // Actualizar las etiquetas en la imagen umbralizada
                    for (const auto& punto : part) {
                        labeled.at<unsigned short>(punto.y, punto.x) = newLabel;
                    }
                    newLabel++;  // Incrementa la etiqueta para la siguiente subregión
                }
            }
        }
        else {
            newRegiones.push_back(region);
            newSizes.push_back(size);

            // Actualizar las etiquetas en la imagen umbralizada
            for (const auto& punto : region) {
                labeled.at<unsigned short>(punto.y, punto.x) = newLabel;
            }
            newLabel++;  // Incrementa la etiqueta para la siguiente región
        }
    }

    regiones = newRegiones;
    sizes = newSizes;
}



/*void umbral2(const cv::Mat& image, cv::Mat& grayscale) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = image;
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    cv::Mat binary;
    double otsu_thresh_val = cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::cout << "Umbral determinado por Otsu: " << otsu_thresh_val << std::endl;

    // Convertir la imagen binaria a 16 bits
    binary.convertTo(grayscale, CV_16U, 65535.0 / 255.0);
}*/

void umbral2(const cv::Mat& image, cv::Mat& grayscale) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Aplicar filtrado de mediana para reducir el ruido
    cv::Mat medianFiltered;
    cv::medianBlur(gray, medianFiltered, 5); // Ajustar el tamaño del kernel según sea necesario

    // Aplicar umbralización de Otsu
    cv::Mat binary;
    double otsu_thresh_val = cv::threshold(medianFiltered, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::cout << "Umbral determinado por Otsu: " << otsu_thresh_val << std::endl;

    // Convertir la imagen binaria a 16 bits
    binary.convertTo(grayscale, CV_16U, 65535.0 / 255.0);
}






using namespace cv;
using namespace std;

Mat ajustarIluminacion(const Mat& src) {
    Mat labImage;
    // Convertir la imagen al espacio de color LAB
    cvtColor(src, labImage, COLOR_BGR2Lab);

    vector<Mat> labPlanes(3);
    // Dividir la imagen en sus canales L, A y B
    split(labImage, labPlanes);

    // Aplicar CLAHE al canal L
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2.0);
    Mat claheImage;
    clahe->apply(labPlanes[0], claheImage);

    // Fusionar los canales de nuevo
    claheImage.copyTo(labPlanes[0]);
    merge(labPlanes, labImage);

    Mat result;
    // Convertir la imagen de nuevo al espacio de color BGR
    cvtColor(labImage, result, COLOR_Lab2BGR);

    // Incrementar un poco el brillo de la imagen
    result.convertTo(result, -1, 1, 50); // Incrementar brillo en 50 unidades

    return result;
}

/*
void corregirIluminacion(const cv::Mat& inputImage, cv::Mat& outputImage) {
    // Convert the image to the YCrCb color space
    cv::Mat ycrcb;
    cvtColor(inputImage, ycrcb, cv::COLOR_BGR2YCrCb);

    // Split the channels
    std::vector<cv::Mat> channels;
    split(ycrcb, channels);

    // Equalize the histogram of the Y channel
    equalizeHist(channels[0], channels[0]);

    // Merge the channels back
    merge(channels, ycrcb);

    // Convert the image back to the BGR color space
    cvtColor(ycrcb, outputImage, cv::COLOR_YCrCb2BGR);
}
*/


//Funci n para dividir las regiones que no sean esf ricas 
void SplitGroups(std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones, std::vector<double>& esfericidades) {

    size_t originalSize = regiones.size(); // Guarda el tama o original del vector

    for (size_t i = 0; i < originalSize; ++i) { // Itera solo sobre las regiones originales

        if (!regiones[i].empty()) {
            float esfericidad_1 = esfericidad(regiones[i], esfericidades);
            std::vector<std::vector<cv::Point>> subRegions2 = kMeans(regiones[i], 2);
            float esfericidad_2 = (esfericidad(subRegions2[0], esfericidades) + esfericidad(subRegions2[1], esfericidades)) / 2.0;
            std::vector<std::vector<cv::Point>> subRegions3 = kMeans(regiones[i], 3);
            float esfericidad_3 = 0.0;
            for (int j = 0; j < subRegions3.size(); j++)
                esfericidad_3 += esfericidad(subRegions3[j], esfericidades);
            esfericidad_3 /= 3.0;
            std::vector<std::vector<cv::Point>> subRegions4 = kMeans(regiones[i], 4);
            float esfericidad_4 = 0.0;
            for (int j = 0; j < subRegions4.size(); j++)
                esfericidad_4 += esfericidad(subRegions4[j], esfericidades);
            esfericidad_4 /= 4.0;

            //falta meter aqu  la comparativa entre 1,2,3 y 4
            if (esfericidad_2 > esfericidad_1) {
                regiones[i] = std::move(subRegions2[0]);
                regiones.push_back(std::move(subRegions2[1]));
            }
            else if (esfericidad_3 > esfericidad_2) {
                regiones[i] = std::move(subRegions3[0]);
                regiones.push_back(std::move(subRegions3[1]));
            }
            else if (esfericidad_4 > esfericidad_3) {
                regiones[i] = std::move(subRegions4[0]);
                regiones.push_back(std::move(subRegions4[1]));
            }
        }
    }
}

double esfericidad(std::vector<cv::Point> region, std::vector<double>& esfericidades) {
    if (region.empty()) {
        return 0.0; // Si la regi n est  vac a, la esfericidad es cero.
    }

    cv::Point centroid(0, 0);
    for (const auto& point : region) {
        centroid += point;
    }
    centroid.x /= region.size();
    centroid.y /= region.size();

    double radioCuadrado = (double)region.size() / 3.141582;
    int puntosDentro = 0;
    for (int i = 0; i < region.size(); i++)
        if (squaredDistance(region[i], centroid) < radioCuadrado)
            puntosDentro++;

    double esfericidad = (double)puntosDentro / region.size();
    esfericidades.push_back(esfericidad); // Almacenar esfericidad en el vector

    return esfericidad;
}


std::vector<std::vector<cv::Point>> kMeans(std::vector<cv::Point> region, int nClusters) {
    std::vector<cv::Point> centroids(nClusters);
    std::vector<std::vector<cv::Point>> subRegions(nClusters);

    bool changed = true;
    int iterations = 100;  // Limitar el n mero de iteraciones

    for (int i = 0; i < nClusters; i++)
        centroids[i] = region[i * region.size() / nClusters];

    while (changed && iterations-- > 0) {
        for (auto& subRegion : subRegions)
            subRegion.clear();  // Limpiar subRegiones antes de la pr xima iteraci n si va a haber una
        changed = false;

        for (auto& point : region) {
            int closestCentroidIdx = 0;
            double closestDist = squaredDistance(point, centroids[0]);

            for (int j = 1; j < nClusters; j++) {
                double dist = squaredDistance(point, centroids[j]);
                if (dist < closestDist) {
                    closestDist = dist;
                    closestCentroidIdx = j;
                }
            }

            subRegions[closestCentroidIdx].push_back(point);
        }

        for (int i = 0; i < nClusters; i++) {
            cv::Point newCentroid(0, 0);
            for (auto& point : subRegions[i]) {
                newCentroid += point;
            }

            if (!subRegions[i].empty()) {
                newCentroid.x /= subRegions[i].size();
                newCentroid.y /= subRegions[i].size();

                if (newCentroid != centroids[i]) {
                    centroids[i] = newCentroid;
                    changed = true;
                }
            }
        }
    }

    return subRegions;
}

double squaredDistance(const cv::Point& p1, const cv::Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    //return std::sqrt(dx * dx + dy * dy);
    return dx * dx + dy * dy; // Eliminando la llamada a sqrt
}



