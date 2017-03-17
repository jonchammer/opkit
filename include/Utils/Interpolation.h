#ifndef INTERPOLATION_H
#define INTERPOLATION_H

namespace opkit
{

// This file includes implementations for several common interpolation
// routines that can be used when manipulating images. Interpolation is
// generally useful when an algorithm produces a real number for an (x, y)
// coordinate.

template <class T>
struct Interpolator
{
    virtual T interpolate(const T* src, const size_t width, const size_t height,
        const T srcX, const T srcY) = 0;
};

// Chooses the closest source pixel to the desired location. Values outside the
// range of the image are set to 0. This is fast, but provides blocky results.
template <class T>
struct NearestNeighborInterpolator : public Interpolator<T>
{
    T interpolate(const T* src, const size_t width, const size_t height,
        const T srcX, const T srcY) override
    {
        int y = (int) (srcY + T{0.5});
        int x = (int) (srcX + T{0.5});

        if (x >= 0 && x < width && y >= 0 && y < height)
            return src[y * width + x];
        else return T{};
    }
};

// Chooses the source pixel based on bilinear interpolation. Values outside the
// range of the image are set to 0. This is slower, but produces better results.
template <class T>
struct BilinearInterpolator : public Interpolator<T>
{
    T interpolate(const T* src, const size_t width, const size_t height,
        const T srcX, const T srcY) override
    {
        // Simple out-of-bounds check
        if (srcX < T{} || srcX >= T{width-1} || srcY < T{} || srcY >= T{height-1})
            return T{};

        // Determine the starting point and the shift amounts
        int x   = (int) srcX;
        int y   = (int) srcY;
        T alpha = srcX - T{x};
        T beta  = srcY - T{y};

        // Isolate the 4 corners
        int base  = y * width + x;
        int base2 = base + width;
        T val1    = src[base];
        T val2    = src[base + 1];
        T val3    = src[base2];
        T val4    = src[base2 + 1];

        // Compute the weighted sum to interpolate
        return val1 * (1.0 - alpha) * (1.0 - beta) +
               val2 * (      alpha) * (1.0 - beta) +
               val3 * (1.0 - alpha) * (      beta) +
               val3 * (      alpha) * (      beta);
    }
};

// Chooses the source pixel based on bilinear interpolation. Values outside the
// range of the image are effectively clamped. This tends to produce more
// consistent results, since there are no artificial black boxes.
template <class T>
struct ClampedBilinearInterpolator : public Interpolator<T>
{
    T interpolate(const T* src, const size_t width, const size_t height,
        T srcX, T srcY) override
    {
        // Clamp check
        if (srcX < T{})
            srcX = T{};
        else if (srcX >= T{width - 1})
            srcX = T{width} - T{1.00001};
        if (srcY < T{})
            srcY = T{};
        else if (srcY >= T{height - 1})
            srcY = T{height} - T{1.00001};

        // Determine the starting point and the shift amounts
        int x   = (int) srcX;
        int y   = (int) srcY;
        T alpha = srcX - T{x};
        T beta  = srcY - T{y};

        // Isolate the 4 corners
        int base  = y * width + x;
        int base2 = base + width;
        T val1    = src[base];
        T val2    = src[base + 1];
        T val3    = src[base2];
        T val4    = src[base2 + 1];

        // Compute the weighted sum to interpolate
        return val1 * (1.0 - alpha) * (1.0 - beta) +
               val2 * (      alpha) * (1.0 - beta) +
               val3 * (1.0 - alpha) * (      beta) +
               val3 * (      alpha) * (      beta);
    }
};

}

#endif
