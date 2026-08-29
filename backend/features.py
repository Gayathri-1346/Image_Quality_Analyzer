import cv2
import numpy as np


def sharpness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()   
    score = 1 - np.exp(-variance / 500)
    return float(np.clip(score, 0, 1))

def exposure_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    dark_pixels = np.sum(gray < 20)
    bright_pixels = np.sum(gray > 235)
    total_pixels = gray.size
    underexposed_ratio = dark_pixels / total_pixels
    overexposed_ratio = bright_pixels / total_pixels
    clipping_ratio = underexposed_ratio + overexposed_ratio
    exposure_score = 1 - min(clipping_ratio * 2, 1)
    brightness_penalty = abs(mean_brightness - 127.5) / 127.5
    score = exposure_score * (1 - 0.5 * brightness_penalty)
    return {
        "exposure_score": float(np.clip(score, 0, 1)),
        "brightness": float(mean_brightness),
        "underexposed_ratio": float(underexposed_ratio),
        "overexposed_ratio": float(overexposed_ratio)
    }

def noise_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred
    noise_level = np.std(residual)
    noise_score = np.exp(-noise_level / 15)
    return {
        "noise_score": float(np.clip(noise_score, 0, 1)),
        "noise_level": float(noise_level)
    }


def contrast_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    score = min(contrast / 64, 1)
    return {
        "contrast_score": float(np.clip(score, 0, 1)),
        "contrast_value": float(contrast)
    }


def saturation_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    score = min(mean_saturation / 128, 1)
    return {
        "saturation_score": float(np.clip(score, 0, 1)),
        "mean_saturation": float(mean_saturation)
    }

def extract_features(image):
    exposure = exposure_features(image)
    noise = noise_features(image)
    contrast = contrast_features(image)
    saturation = saturation_features(image)
    features = {
        "sharpness": sharpness_score(image),
        "exposure": exposure["exposure_score"],
        "brightness": exposure["brightness"],
        "underexposed_ratio": exposure["underexposed_ratio"],
        "overexposed_ratio": exposure["overexposed_ratio"],
        "noise": noise["noise_score"],
        "noise_level": noise["noise_level"],
        "contrast": contrast["contrast_score"],
        "contrast_value": contrast["contrast_value"],
        "saturation": saturation["saturation_score"],
        "mean_saturation": saturation["mean_saturation"]
    }

    return features