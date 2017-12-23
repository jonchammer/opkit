#include <iostream>
#include "il.h"
#include "opkit/opkit.h"

using std::string;
using opkit::Tensor;

class Channel
{
public:

    // Constructors
    Channel(const size_t width, const size_t height) :
        mBuffer(new uint8_t[width * height]),
        mWidth(width), mHeight(height) {};

    Channel() :
        mBuffer(nullptr), mWidth(0), mHeight(0) {}

    Channel(const Channel& other) :
        mBuffer(new uint8_t[other.mWidth * other.mHeight]),
        mWidth(other.mWidth), mHeight(other.mHeight)
    {
        std::copy(other.mBuffer, other.mBuffer + mWidth * mHeight, mBuffer);
    }

    Channel(Channel&& other) :
        mBuffer(other.mBuffer),
        mWidth(other.mWidth),
        mHeight(other.mHeight)
    {
        other.mBuffer = nullptr;
    }

    ~Channel()
    {
        if (mBuffer != nullptr)
            delete[] mBuffer;
        mBuffer = nullptr;
    }

    Channel& operator=(const Channel& other)
    {
        if (this != &other)
        {
            if (mBuffer != nullptr) delete[] mBuffer;
            mWidth  = other.mWidth;
            mHeight = other.mHeight;
            mBuffer = new uint8_t[mWidth * mHeight];
            std::copy(other.mBuffer, other.mBuffer + mWidth * mHeight, mBuffer);
        }
        return *this;
    }

    Channel& operator=(Channel&& other)
    {
        if (this != &other)
        {
            if (mBuffer != nullptr) delete[] mBuffer;
            mWidth  = other.mWidth;
            mHeight = other.mHeight;
            mBuffer = other.mBuffer;
            other.mBuffer = nullptr;
        }
        return *this;
    }

    bool save(const string& filename) const
    {
        // Create a new IL Image
        ilInit();
        ILuint id;
        ilGenImages(1, &id);
        ilBindImage(id);

        // Copy to the IL Image
        ilTexImage(mWidth, mHeight, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, mBuffer);
        //ilTexImage(mWidth, mHeight, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, buffer);

        // Save the image
        ilEnable(IL_FILE_OVERWRITE);
        return ilSaveImage(filename.c_str());
    }

    uint8_t get(const size_t x, const size_t y) const
    {
        return mBuffer[y * mWidth + x];
    }

    void set(const size_t x, const size_t y, const uint8_t value,
        const bool yFlip = false)
    {
        if (yFlip)
            mBuffer[(mHeight - 1 - y) * mWidth + x] = value;
        else mBuffer[y * mWidth + x] = value;
    }

    void fill(const uint8_t value)
    {
        std::fill(mBuffer, mBuffer + mWidth * mHeight, value);
    }

    void copyBlock(const Channel& orig,
        const size_t origX,     const size_t origY,
        const size_t origWidth, const size_t origHeight,
        const size_t destX,     const size_t destY)
    {
        for (int y = 0; y < origHeight; ++y)
        {
            for (int x = 0; x < origWidth; ++x)
            {
                //cout << (int) get(destX + x, destY + y) << " ";
                uint8_t val = orig.get(origX + x, origY + y);
                set(destX + x, destY + y, val);
                //cout << (int) get(destX + x, destY + y) << "; ";
            }
        }
    }

    size_t width()  const { return mWidth;  }
    size_t height() const { return mHeight; }

private:
    uint8_t* mBuffer;
    size_t mWidth;
    size_t mHeight;
};

class Image
{
public:

    void addChannel(Channel& channel)
    {
        mChannels.push_back(&channel);
    }

    bool save(const string& filename)
    {
        if (mChannels.size() == 1)
            return mChannels[0]->save(filename);
        else
        {
            std::cerr << "Only 1 channel is currently supported." << std::endl;
            return false;
        }
    }

private:
    std::vector<Channel*> mChannels;
};

template <class T>
Channel channelFrom1DTensor(const Tensor<T>& data, const size_t width, const size_t height)
{
    Channel res(width, height);
    auto it = data.begin();
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            res.set(x, y, *it * 255, true);
            ++it;
        }
    }
    return res;
}

template <class T>
Channel channelFrom2DTensor(const Tensor<T>& data)
{
    const size_t width  = data.shape(1);
    const size_t height = data.shape(0);

    Channel res(width, height);
    auto it = data.begin();
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            res.set(x, y, *it * 255, true);
            ++it;
        }
    }
    return res;
}

template <class T>
bool plotSingle(const string& filename, const Tensor<T>& values,
    const size_t width, const size_t height)
{
    Image result;
    if (values.rank() == 1)
        result.addChannel(channelFrom1DTensor(values, width, height));
    else if (values.rank() == 2)
        result.addChannel(channelFrom2DTensor(values));

    return result.save(filename);
}

template <class T>
bool plotGrid(const string& filename, const Tensor<T>& images,
    const size_t rows,            const size_t cols,
    const size_t individualWidth, const size_t individualHeight,
    const size_t paddingX = 0,    const size_t paddingY = 0,
    const size_t borderX = 0,     const size_t borderY = 0)
{
    const size_t totalWidth  = cols * individualWidth  + (cols - 1) * paddingX + 2 * borderX;
    const size_t totalHeight = rows * individualHeight + (rows - 1) * paddingY + 2 * borderY;

    Channel canvas(totalWidth, totalHeight);
    canvas.fill(255);

    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            // When we don't have enough images to fill the grid, leave it blank
            if (r * cols + c >= images.shape(0))
                continue;

            // Pick the appropriate image
            Tensor<T> image = select(images, 0, r * cols + c);

            // Turn it into a channel
            Channel temp;
            if (image.rank() == 1)
                temp = channelFrom1DTensor(image, individualWidth, individualHeight);
            else if (image.rank() == 2)
                temp = channelFrom2DTensor(image);

            // Copy this channel into the correct spot on the canvas
            const size_t destX = c              * (individualWidth  + paddingX) + borderX;
            const size_t destY = (rows - 1 - r) * (individualHeight + paddingY) + borderY;
            canvas.copyBlock(temp, 0, 0, individualWidth, individualHeight, destX, destY);
        }
    }

    Image result;
    result.addChannel(canvas);
    return result.save(filename);
}
