#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <complex>


using namespace std;


string toString(char *c) {
    stringstream ss;
    string s;
    ss << c;
    ss >> s;
    return s;
}


int toInt(char *c) {
    stringstream ss;
    int i;
    ss << c;
    ss >> i;
    return i;
}


float toFloat(char *c) {
    stringstream ss;
    float f;
    ss << c;
    ss >> f;
    return f;
}


double toDouble(char *c) {
    stringstream ss;
    double d;
    ss << c;
    ss >> d;
    return d;
}


// (2 + 4 + 4 + 4) = 14 bytes
#pragma pack(push, 1)
class BmpHeader {

public:
    char signature[2];
	uint32_t filesize;           // total size of bitmap file
	uint16_t reserved1;
	uint16_t reserved2;
	uint32_t pixelDataOffset;    // Start position of pixel data (bytes from the beginning of the file)

    // default constructor
    BmpHeader(){};

    BmpHeader(char s1, char s2, uint32_t fsize, uint16_t r1, uint32_t r2, uint32_t offset) {
        signature[0] = s1;
        signature[1] = s2;
        filesize = fsize;
        reserved1 = r1;
        reserved2 = r2;
        pixelDataOffset = offset;
    }
};


// (4 + 4 + 4 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 4) = 40 bytes
class BmpInfoHeader {

public:
	uint32_t sizeOfInfoHeader;
    //   (if positive, bottom-up, with origin in lower left corner)
    //   (if negative, top-down, with origin in upper left corner)
	int32_t width;
	int32_t height;

	uint16_t ColorPlanes;   // awlays 1
	uint16_t bitsPerPixel;  // 24 bits

	uint32_t compressionMethod; // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
	uint32_t rawBitmapDataSize; // 0 - for uncompressed images

	int32_t xPixelPerMeter{0};
	int32_t yPixelPerMeter{0};

	uint32_t colorTableEntries{0};
	uint32_t importantColors{0};

    // default constructor
    BmpInfoHeader(){};

    BmpInfoHeader(uint32_t s, int32_t w, int32_t h, uint16_t cp, uint16_t bpp, uint32_t cpm, uint32_t rbms, int32_t xpp, int32_t ypp, uint32_t ct, uint32_t ic) {
        sizeOfInfoHeader = s;
        width = w;
        height = h;
        ColorPlanes = cp;
        bitsPerPixel = bpp;
        compressionMethod = cpm;
        rawBitmapDataSize = rbms;
        xPixelPerMeter = xpp;
        yPixelPerMeter = ypp;
        colorTableEntries = ct;
        importantColors = ic;
    }
};
#pragma pack(pop)


class BMP {

public:
    BmpHeader file_header;
    BmpInfoHeader bmp_info_header;
    int argc_bmp;
    char **argv_bmp;
    string mode;    // image manipulation mode
    float ratio;
    float beta;
    float kernel_size;
    string filename1;
    string filename2;
    string filename3;
    string ofilename;   // for mode = psnr, this file serves for the other input image

    // this data with padding information
    // vector<uint8_t> data;   // store picture data -- in (B, G, R) order or (B, G, R, A)

    // constructor -- for reading file
    BMP(int argc, char **argv) {
        argc_bmp = argc;
        argv_bmp = argv;
        mode = toString(argv_bmp[1]);
    }

    // read and write
    vector<uint8_t> bmp_read(const string fname);
    void bmp_write(const string fname, vector<uint8_t>& outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header);

    // show messages
    void showBmpHeader(BmpHeader &header);
    void showBmpInfoHeader(BmpInfoHeader &header);

    // call manipulate functions
    void ImageProcessing() {
        if(mode == "-id") {
            filename1 = toString(argv_bmp[2]);
            ofilename = toString(argv_bmp[3]);
            IdentityOut();
        } else if(mode == "-db") {
            filename1 = toString(argv_bmp[2]);
            ofilename = toString(argv_bmp[3]);
            BlindDebluring(toDouble(argv_bmp[4]), toInt(argv_bmp[5]), toInt(argv_bmp[6]), toDouble(argv_bmp[7]));
        } else if(mode == "-psnr") {
            filename1 = toString(argv_bmp[2]);
            filename2 = toString(argv_bmp[3]);
            psnr();
        } else if(mode == "-ndb") {
            filename1 = toString(argv_bmp[2]); // input1
            filename2 = toString(argv_bmp[3]); // input1_ori
            filename3 = toString(argv_bmp[4]); // input2
            ofilename = toString(argv_bmp[5]); // output2
            NonBlindDebluring(toDouble(argv_bmp[6]));
        }

        // default
        else {
            IdentityOut();
        }
        return;
    }

private:
    int width;
    int height;
    int bitsPerPixel;
    int real_image_size; // size with padding

    // eeip_hw1 part
    void write_headers_and_data(ofstream &outfile, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header);
    void IdentityOut();

    // Image BlindDebluring
    // FFT, IFFT
    // Wiener function to do reverse filtering
    void fft1d(vector< complex<double> > &signal, int inverse);             // fft -> inverse = 1, ifft -> inverse = -1
    void fft2d(vector< vector< complex<double> > > &image, int inverse);    // fft -> inverse = 1, ifft -> inverse = -1
    void BlindDebluring(double sigma, int kernel_size, int kernel_size_blur, double eps);
    void NonBlindDebluring(double eps);

    // Compute PSNR
    void psnr();

};


void BMP::showBmpHeader(BmpHeader &header) {
    cout << "===============================================================" << endl;
    cout << "Bmp Header" << endl;
    stringstream ss;
    string s;
    ss << "0x" << std::hex << (int)header.signature[1] << (int)header.signature[0];
    ss >> s;
    cout << "file type: " << s << endl;
    cout << "file size: " << header.filesize << endl;
    cout << "reservedByte: " << std::dec << header.reserved1 << header.reserved2 << endl;
    cout << "Pixel Offset: " << header.pixelDataOffset << endl;
    cout << "===============================================================" << endl;
}


void BMP::showBmpInfoHeader(BmpInfoHeader &header) {
    cout << "===============================================================" << endl;
    cout << "Bmp Info Header" << endl;
    cout << "sizeOfInfoHeader: " << header.sizeOfInfoHeader << endl;
    cout << "width: " << header.width << endl;
    cout << "height: " << header.height << endl;
    cout << "numberOfColorPlanes: " << header.ColorPlanes << endl;
    cout << "bitsPerPixel: " << header.bitsPerPixel << endl;
    cout << "compressionMethod: " << header.compressionMethod << endl;
    cout << "rawBitmapDataSize: " << header.rawBitmapDataSize << endl;
    cout << "xPixelPerMeter: " << header.xPixelPerMeter << endl;
    cout << "yPixelPerMeter: " << header.yPixelPerMeter << endl;
    cout << "colorTableEntries: " << header.colorTableEntries << endl;
    cout << "importantColors: " << header.importantColors << endl;
    cout << "===============================================================" << endl;
}


void BMP::write_headers_and_data(ofstream &outfile, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header) {
    cout << "Show write header:" << endl;
    if(!custom_header) {
        outfile.write((const char*)&file_header, sizeof(file_header));
        outfile.write((const char*)&bmp_info_header, sizeof(bmp_info_header));
        outfile.write((const char*)outdata.data(), outdata.size());
        showBmpHeader(file_header);
        showBmpInfoHeader(bmp_info_header);
    } else {
        outfile.write((const char*)&BH, sizeof(BH));
        outfile.write((const char*)&BIH, sizeof(BIH));
        outfile.write((const char*)outdata.data(), outdata.size());
        showBmpHeader(BH);
        showBmpInfoHeader(BIH);
    }
}


void BMP::bmp_write(const string fname, vector<uint8_t> &outdata, BmpHeader BH, BmpInfoHeader BIH, bool custom_header) {
    ofstream outfile(fname, ios_base::binary);
    if(outfile) {
        write_headers_and_data(outfile, outdata, BH, BIH, custom_header);
    } else {
        cout << "Error! File not correctly written!" << endl;
    }
    cout << "File written successfully!" << endl;
    outfile.close();
}


vector<uint8_t> BMP::bmp_read(const string fname) {
    ifstream infile(fname, ios_base::binary);
    vector<uint8_t> data;

    if(infile) {
        infile.read((char*)&file_header, sizeof(file_header));
        if(file_header.signature[0] != 'B' || file_header.signature[1] != 'M') {
            cerr << "Error: Not a valid BMP file." << endl;
            exit(1);
        }
        infile.read((char*)&bmp_info_header, sizeof(bmp_info_header));

        // store public variable
        width = bmp_info_header.width;
        height = bmp_info_header.height;
        bitsPerPixel = bmp_info_header.bitsPerPixel;

        // compute important data
        int channels = (bitsPerPixel / 8);
        int real_row_width = ceil(width * channels / 4.0) * 4;
        real_image_size = height * real_row_width;

        // jump to the pixel data location
        infile.seekg(file_header.pixelDataOffset, infile.beg);

        // resize the pixel data vector
        data.resize(real_image_size);

        // store data
        infile.read((char*)data.data(), data.size());
    }
    // force pixel offset to 14 + 40 bytes
    file_header.pixelDataOffset = 54;
    // force bmp info header to 40 bytes
    bmp_info_header.sizeOfInfoHeader = 40;
    file_header.filesize = real_image_size + sizeof(BmpHeader) + sizeof(BmpInfoHeader);

    // cout << "Show read header:" << endl;
    // showBmpHeader(file_header);
    // showBmpInfoHeader(bmp_info_header);
    infile.close();
    return data;
}

// this function might only change the info header size and the filesize
void BMP::IdentityOut() {
    vector<uint8_t> data;
    data = bmp_read(filename1);
    string ofname = ofilename;
    bmp_write(ofname, data, file_header, bmp_info_header, false);
    return;
}


int ReverseBin(int a, int n) {
    int ret = 0;
    for(int i=0; i<n; i++) {
        if(a & (1 << i))
            ret |= (1 << (n - 1 - i));
    }
    return ret;
}


void BMP::fft1d(vector< complex<double> > &signal, int inverse=1) {
    // where signal is a row or a column of the image with single channel    
    const int N = signal.size();    // size of data
    // prebuild the butterfly shape index
    int idx;
    vector<complex<double>> temp(N);
    for(int i=0; i<N; i++) {
        idx = ReverseBin(i, log2(N));
        temp[i] = signal[idx];
    }

    // precompute WN table
    vector<complex<double>> wn(N/2);
    double angle = 2 * M_PI / N;
    for(int i=0; i<N/2; i++) {
        wn[i] = {cos(angle * i), inverse * (-1) * sin(angle * i)};
    }

    int id0, id1;
    complex<double> tmp;
    for(int steplength=2; steplength<=N; steplength*=2) {
        for(int step=0; step<N/steplength; step++) {
            for(int i=0; i<steplength/2; i++) {
                id0 = steplength * step + i;
                id1 = steplength * step + i + steplength / 2;
                tmp = temp[id1] * wn[N / steplength * i];
                temp[id1] = temp[id0] - tmp;
                temp[id0] = temp[id0] + tmp;
            }
        }
    }

    for(int i=0; i<N; i++) {
        if(inverse == -1) {
            signal[i] = temp[i] / complex<double>(N, 0);
        }
        else {
            signal[i] = temp[i];
        }
    }
    return;
}


void BMP::fft2d(vector< vector< complex<double> > > &image, int inverse=1) {
    // where image is a plane of one channel of the image
    int rows = image.size();
    int cols = image[0].size();

    for(int y=0; y<rows; y++) {
        vector<complex<double>> temp(cols);
        // get temp row data
        for(int x=0; x<cols; x++) {
            temp[x] = image[y][x];
        }

        // row fft
        fft1d(temp, inverse);

        // write back to image
        for(int x=0; x<cols; x++) {
            image[y][x] = temp[x];
        }
    }

    for(int x=0; x<cols; x++) {
        vector<complex<double>> temp(rows);
        // get temp col data
        for(int y=0; y<rows; y++) {
            temp[y] = image[y][x];
        }

        // col fft
        fft1d(temp, inverse);

        // write back to image
        for(int y=0; y<rows; y++) {
            image[y][x] = temp[y];
        }
    }

    return;
}


void BMP::BlindDebluring(double sigma, int kernel_size, int kernel_size_blur, double eps=0.005) {
    vector<uint8_t> data(real_image_size);
    data = bmp_read(filename1);

    string ofname = ofilename;

    vector<uint8_t> data_db;
    data_db.resize(real_image_size);
    int channels = bitsPerPixel / 8;
    int real_row_width = ceil(width * channels / 4.0) * 4;

    int pad_height, pad_width;
    pad_height = pow(2, ceil(log2(height)));
    pad_width = pow(2, ceil(log2(width)));

    int N = kernel_size;
    int padding = N / 2;
    complex<double> GaussianBlurKernel[N][N] = {};

    // create Gaussian blur kernel
    double r;
    complex<double> sum(0);
    double dev = 2.0 * sigma * sigma;
    
    // generate Gaussian blur kernel
    for(int x=-padding; x<padding+1; x++) {
        for(int y=-padding; y<padding+1; y++) {
            r = (x * x + y * y);
            GaussianBlurKernel[x+padding][y+padding] = (exp((-1) * r / dev)) / (M_PI * dev);
            sum += GaussianBlurKernel[x+padding][y+padding];
        }
    }
    // normalize kernel
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            GaussianBlurKernel[i][j] /= sum;
        }
    }

    // create Motion blur kernel (degree = 45)
    int Nb = kernel_size_blur;
    int padding_b = Nb / 2;
    complex<double> MotionBlurKernel[Nb][Nb] = {};

    for(int i=0; i<Nb; i++) {
        for(int j=0; j<Nb; j++) {
            if(i == j)
                MotionBlurKernel[i][j] = 1.0 / double(Nb);
            else
                MotionBlurKernel[i][j] = 0;
        }
    }

    vector<vector<complex<double>>> Bplane(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Gplane(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Rplane(pad_height, vector<complex<double>>(pad_width, 0));

    vector<vector<complex<double>>> GaussianBlurKernelN_F(pad_height+N, vector<complex<double>>(pad_width+N, 0));
    vector<vector<complex<double>>> MotionBlurKernelNb_F(pad_height+Nb, vector<complex<double>>(pad_width+Nb, 0));

    for(int i=-padding; i<padding+1; i++) {
        for(int j=-padding; j<padding+1; j++) {
            GaussianBlurKernelN_F[i + padding][j + padding] = GaussianBlurKernel[i + padding][j + padding];
            GaussianBlurKernelN_F[i + padding][j + padding + pad_width] = GaussianBlurKernel[i + padding][j + padding];
            GaussianBlurKernelN_F[i + padding + pad_height][j + padding] = GaussianBlurKernel[i + padding][j + padding];
            GaussianBlurKernelN_F[i + padding + pad_height][j + padding + pad_width] = GaussianBlurKernel[i + padding][j + padding];
        }
    }

    for(int i=-padding_b; i<padding_b+1; i++) {
        for(int j=-padding_b; j<padding_b+1; j++) {
            MotionBlurKernelNb_F[i + padding_b][j + padding_b] = MotionBlurKernel[i + padding_b][j + padding_b];
            MotionBlurKernelNb_F[i + padding_b][j + padding_b + pad_width] = MotionBlurKernel[i + padding_b][j + padding_b];
            MotionBlurKernelNb_F[i + padding_b + pad_height][j + padding_b] = MotionBlurKernel[i + padding_b][j + padding_b];
            MotionBlurKernelNb_F[i + padding_b + pad_height][j + padding_b + pad_width] = MotionBlurKernel[i + padding_b][j + padding_b];
        }
    }

    vector<vector<complex<double>>> GaussianBlurKernel_F(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> MotionBlurKernel_F(pad_height, vector<complex<double>>(pad_width, 0));

    for(int i=padding; i<pad_height+padding; i++) {
        for(int j=padding; j<pad_width+padding; j++) {
            GaussianBlurKernel_F[i-padding][j-padding] = GaussianBlurKernelN_F[i][j];
            MotionBlurKernel_F[i-padding][j-padding] = MotionBlurKernelNb_F[i][j];
        }
    }

    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width + x * channels;
            Bplane[y][x] = {(double)data[offset + 0], 0};
            Gplane[y][x] = {(double)data[offset + 1], 0};
            Rplane[y][x] = {(double)data[offset + 2], 0};
        }
    }

    // do fft
    fft2d(Bplane, 1);
    fft2d(Gplane, 1);
    fft2d(Rplane, 1);

    fft2d(GaussianBlurKernel_F, 1);
    fft2d(MotionBlurKernel_F, 1);

    for(int y=0; y<pad_height; ++y) {
        for(int x=0; x<pad_width; ++x) {
            complex<double> H = GaussianBlurKernel_F[y][x] * MotionBlurKernel_F[y][x];
            complex<double> wf = (1.0 / H) * (pow(abs(H), 2) / (pow(abs(H), 2) + eps));
            Bplane[y][x] = Bplane[y][x] * wf;
            Gplane[y][x] = Gplane[y][x] * wf;
            Rplane[y][x] = Rplane[y][x] * wf;
        }
    }

    // do ifft
    fft2d(Bplane, -1);
    fft2d(Gplane, -1);
    fft2d(Rplane, -1);

    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width + x * channels;
            data_db[offset + 0] = (uint8_t)max(min(abs(Bplane[y][x]), 255.0), 0.0);
            data_db[offset + 1] = (uint8_t)max(min(abs(Gplane[y][x]), 255.0), 0.0);
            data_db[offset + 2] = (uint8_t)max(min(abs(Rplane[y][x]), 255.0), 0.0);
        }
    }

    bmp_write(ofname, data_db, file_header, bmp_info_header, false);
    return;
}


void BMP::psnr() {
    double psnrB = 0, psnrG = 0, psnrR = 0, psnr;

    vector<uint8_t> img1;
    vector<uint8_t> img2;

    img1 = bmp_read(filename1);
    int channels_img1 = bitsPerPixel / 8;
    int real_row_width_img1 = ceil(width * channels_img1 / 4.0) * 4;

    img2 = bmp_read(filename2);
    int channels_img2 = bitsPerPixel / 8;
    int real_row_width_img2 = ceil(width * channels_img2 / 4.0) * 4;

    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            int offset_img1 = y * real_row_width_img1 + x * channels_img1;
            int offset_img2 = y * real_row_width_img2 + x * channels_img2;
            psnrB += pow(abs((int)img1[offset_img1 + 0] - (int)img2[offset_img2 + 0]), 2);
            psnrG += pow(abs((int)img1[offset_img1 + 1] - (int)img2[offset_img2 + 1]), 2);
            psnrR += pow(abs((int)img1[offset_img1 + 2] - (int)img2[offset_img2 + 2]), 2);
        }
    }

    // reference method
    psnrB /= (3 * height * width);
    psnrG /= (3 * height * width);
    psnrR /= (3 * height * width);
    psnr = 10 * log10(pow(255, 2) / psnrB) + 10 * log10(pow(255, 2) / psnrG) + 10 * log10(pow(255, 2) / psnrR);

    // wiki method
    // psnr = 10 * log10(pow(255, 2) / ((psnrB + psnrG + psnrR) / (3 * width * height)));

    cout << "PSNR : " << psnr << " db" << endl;
    return;
}


void BMP::NonBlindDebluring(double eps) {
    vector<uint8_t> img1;
    vector<uint8_t> img2;
    vector<uint8_t> img3;

    img1 = bmp_read(filename1);
    int channels_img1 = bitsPerPixel / 8;
    int real_row_width_img1 = ceil(width * channels_img1 / 4.0) * 4;

    img2 = bmp_read(filename2);
    int channels_img2 = bitsPerPixel / 8;
    int real_row_width_img2 = ceil(width * channels_img2 / 4.0) * 4;

    int pad_height, pad_width;
    pad_height = pow(2, ceil(log2(height)));
    pad_width = pow(2, ceil(log2(width)));

    vector<vector<complex<double>>> Bplane_img1(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Gplane_img1(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Rplane_img1(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Bplane_img2(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Gplane_img2(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Rplane_img2(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> estimatedKernelB(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> estimatedKernelG(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> estimatedKernelR(pad_height, vector<complex<double>>(pad_width, 0));

    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            int offset_img1 = y * real_row_width_img1 + x * channels_img1;
            int offset_img2 = y * real_row_width_img2 + x * channels_img2;
            Bplane_img1[y][x] = {(double)img1[offset_img1 + 0], 0};
            Gplane_img1[y][x] = {(double)img1[offset_img1 + 1], 0};
            Rplane_img1[y][x] = {(double)img1[offset_img1 + 2], 0};
            Bplane_img2[y][x] = {(double)img2[offset_img2 + 0], 0};
            Gplane_img2[y][x] = {(double)img2[offset_img2 + 1], 0};
            Rplane_img2[y][x] = {(double)img2[offset_img2 + 2], 0};
        }
    }

    // do fft
    fft2d(Bplane_img1, 1);
    fft2d(Gplane_img1, 1);
    fft2d(Rplane_img1, 1);
    fft2d(Bplane_img2, 1);
    fft2d(Gplane_img2, 1);
    fft2d(Rplane_img2, 1);

    for(int y=0; y<pad_height; y++) {
        for(int x=0; x<pad_width; x++) {
            estimatedKernelB[y][x] = pow(abs(Bplane_img1[y][x]), 2) / (pow(abs(Bplane_img2[y][x]), 2));
            estimatedKernelG[y][x] = pow(abs(Gplane_img1[y][x]), 2) / (pow(abs(Gplane_img2[y][x]), 2));
            estimatedKernelR[y][x] = pow(abs(Rplane_img1[y][x]), 2) / (pow(abs(Rplane_img2[y][x]), 2));
            // estimatedKernelB[y][x] = Bplane_img2[y][x] / (Bplane_img1[y][x] + 1e-8);
            // estimatedKernelG[y][x] = Gplane_img2[y][x] / (Gplane_img1[y][x] + 1e-8);
            // estimatedKernelR[y][x] = Rplane_img2[y][x] / (Rplane_img1[y][x] + 1e-8);
        }
    }

    // img3
    img3 = bmp_read(filename3);
    vector<uint8_t> data_ndb;
    data_ndb.resize(real_image_size);
    int channels_img3 = bitsPerPixel / 8;
    int real_row_width_img3 = ceil(width * channels_img3 / 4.0) * 4;

    pad_height = pow(2, ceil(log2(height)));
    pad_width = pow(2, ceil(log2(width)));

    vector<vector<complex<double>>> Bplane_img3(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Gplane_img3(pad_height, vector<complex<double>>(pad_width, 0));
    vector<vector<complex<double>>> Rplane_img3(pad_height, vector<complex<double>>(pad_width, 0));

    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            int offset_img3 = y * real_row_width_img3 + x * channels_img3;
            Bplane_img3[y][x] = {(double)img3[offset_img3 + 0], 0};
            Gplane_img3[y][x] = {(double)img3[offset_img3 + 1], 0};
            Rplane_img3[y][x] = {(double)img3[offset_img3 + 2], 0};
        }
    }

    fft2d(Bplane_img3, 1);
    fft2d(Gplane_img3, 1);
    fft2d(Rplane_img3, 1);

    for(int i=0; i<pad_height; i++) {
        for(int j=0; j<pad_width; j++) {
            Bplane_img3[i][j] = Bplane_img3[i][j] / estimatedKernelB[i%512][j%512] * (pow(estimatedKernelB[i%512][j%512], 2) / (pow(estimatedKernelB[i%512][j%512], 2) + eps));
            Gplane_img3[i][j] = Gplane_img3[i][j] / estimatedKernelG[i%512][j%512] * (pow(estimatedKernelG[i%512][j%512], 2) / (pow(estimatedKernelG[i%512][j%512], 2) + eps));
            Rplane_img3[i][j] = Rplane_img3[i][j] / estimatedKernelR[i%512][j%512] * (pow(estimatedKernelR[i%512][j%512], 2) / (pow(estimatedKernelR[i%512][j%512], 2) + eps));
        }
    }

    fft2d(Bplane_img3, -1);
    fft2d(Gplane_img3, -1);
    fft2d(Rplane_img3, -1);

    for(int y=0; y<height; ++y) {
        for(int x=0; x<width; ++x) {
            int offset = y * real_row_width_img3 + x * channels_img3;
            data_ndb[offset + 0] = (uint8_t)max(min(abs(Bplane_img3[y][x]), 255.0), 0.0);
            data_ndb[offset + 1] = (uint8_t)max(min(abs(Gplane_img3[y][x]), 255.0), 0.0);
            data_ndb[offset + 2] = (uint8_t)max(min(abs(Rplane_img3[y][x]), 255.0), 0.0);
        }
    }

    bmp_write(ofilename, data_ndb, file_header, bmp_info_header, false);
    return;
}


int main(int argc, char** argv) {

    // constructor
    BMP bmp(argc, argv);

    // do image processing
    bmp.ImageProcessing();

    return 0;
}


