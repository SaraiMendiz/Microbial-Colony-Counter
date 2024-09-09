#include "bmp.h"
#include <numeric>  // Para std::accumulate

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_with_circle> <thresholded_image>" << std::endl;
        return 1;
    }

    std::string image_with_circle_path = argv[1];
    std::string thresholded_image_path = argv[2];

    std::cout << "Image with circle path: " << image_with_circle_path << std::endl;
    std::cout << "Thresholded image path: " << thresholded_image_path << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Cargar la imagen con el círculo
    cv::Mat image = cv::imread(image_with_circle_path);
    if (image.empty())
    {
        std::cout << "No se ha podido cargar la imagen con el círculo." << std::endl;
        return 1;
    }

    // Cargar la imagen umbralizada
    cv::Mat labeled = cv::imread(thresholded_image_path, cv::IMREAD_GRAYSCALE);
    if (labeled.empty())
    {
        std::cout << "No se ha podido cargar la imagen umbralizada." << std::endl;
        return 1;
    }
    labeled.convertTo(labeled, CV_16U, 65535.0 / 255.0);

    std::vector<int> labeledArray;//aqui se guardan el numero de pixeles que hay en cada region
    std::vector<std::vector<cv::Point>> regiones; // Listados de píxeles pertenecientes a una regiones junto a sus coordenadas
    std::vector<double> esfericidades;

    detectarCirculoMasGrande(image); // Detectamos el círculo más grande
    etiquetarRegiones(labeled, labeledArray, regiones); // Etiquetamos las regiones y a la vez guardamos en el array el número de píxeles que hay en cada región
    descartarRegiones(labeled, 10, labeledArray, regiones); 
    filtrarRegionesDentroDelCirculo(regiones); // Filtramos las regiones que se encuentren dentro del círculo más grande

    // Parámetros para dividirRegionSiEsGrande
    double k_start = 1.30;
    double k_step = 0.05;
    int iterations = 20;

    // Vector para almacenar los resultados de las iteraciones
    std::vector<int> region_counts;

    for(int i = 0; i < iterations; i++){
        // Copia temporal de las regiones y labeledArray
        std::vector<std::vector<cv::Point>> temp_regiones = regiones; 
        std::vector<int> temp_labeledArray = labeledArray;
        dividirRegionSiEsGrande(temp_regiones, temp_labeledArray, labeled, k_start + i * k_step);
        region_counts.push_back(temp_regiones.size());
    }

    // Calcular la media de los conteos de colonias
    double mean_count = std::accumulate(region_counts.begin(), region_counts.end(), 0.0) / iterations;

    pintarRegiones(image, regiones);
    cv::imwrite("imagenFinal.bmp", image);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Tiempo total: " << duration.count() << " milisegundos" << std::endl;
    std::cout << "Bacterias totales (media): " << mean_count << std::endl;

    return 0;
}



/*
int main()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    cv::Mat image = cv::imread("/Users/saraimendizsal/Desktop/TFG/Test4.bmp");

    if (image.empty())
    {
        std::cout << "No se ha podido cargar la imagen." << std::endl;
        return 1;
    }
    double threshold = 75000; //obtenemos el valor umbral pasado como argumento 75000
    cv::Mat labeled(image.size(), CV_16U); //creamos una imagen en escala de grises

    umbral2(image, labeled);
    cv::imwrite("imagenGrises.bmp", labeled);

    std::vector<int> labeledArray;
    std::vector<std::vector<cv::Point>> regiones; //listados de pixeles pertenecientes a una regiones junto a sus coordenadas
    std::vector<double> esfericidades;

    detectarCirculoMasGrande(image); //Detectamos el c rculo m s grande
    etiquetarRegiones(labeled, labeledArray, regiones); //etiquetamos las regiones y a la vez guardamos en el array el numero de pixeles que hay en cada region
    //for (int i = 0; i < 4; i++)
    descartarRegiones(labeled, 10, labeledArray, regiones); 
    //descartarRegiones2(labeled, labeledArray, regiones);

    filtrarRegionesDentroDelCirculo(regiones); //filtramos las regiones que se encuentren dentro del c�rculo m�s grande
    

    //SplitGroups(labeledArray, regiones, esfericidades);
    dividirRegionSiEsGrande(regiones, labeledArray, 2.15);


    pintarRegiones(image, regiones);
    std::vector<cv::Vec3f> circulosDetectados;
    cv::imwrite("imagenFinal.bmp", image);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Tiempo total: " << duration.count() << " milisegundos" << std::endl;
    std::cout << "Bacterias totales: " << regiones.size() << std::endl;

    return 0;
}
*/












