"use client"

import type React from "react"
import { useRef, useState, useCallback } from "react"
import { Upload, ImageIcon, Sparkles, AlertCircle, CheckCircle2, FileImage, Leaf } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface ImageUploadProps {
  onImageSelect: (file: File) => void
  isVerifying?: boolean
}

export const ImageUpload: React.FC<ImageUploadProps> = ({ onImageSelect, isVerifying = false }) => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [isValidating, setIsValidating] = useState(false)

  const validateFile = useCallback((file: File): string | null => {
    // File type validation
    if (!file.type.startsWith("image/")) {
      return "Please select a valid image file (JPG, PNG, JPEG, WebP)"
    }

    // File size validation (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      return "File size must be less than 10MB"
    }

    // File size minimum (to ensure quality)
    if (file.size < 5 * 1024) {
      return "File size too small. Please select a higher quality image"
    }

    return null
  }, [])

  const processFile = useCallback(
    async (file: File) => {
      setIsValidating(true)
      setValidationError(null)
      setUploadProgress(0)

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 100)

      // Validate file
      const error = validateFile(file)
      if (error) {
        setValidationError(error)
        setIsValidating(false)
        clearInterval(progressInterval)
        setUploadProgress(0)
        return
      }

      // Simulate processing time
      await new Promise((resolve) => setTimeout(resolve, 800))

      setUploadProgress(100)
      setTimeout(() => {
        setIsValidating(false)
        onImageSelect(file)
      }, 200)
    },
    [validateFile, onImageSelect],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragOver(false)

      const files = e.dataTransfer.files
      if (files.length > 0) {
        processFile(files[0])
      }
    },
    [processFile],
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files && files.length > 0) {
        processFile(files[0])
      }
    },
    [processFile],
  )

  const handleClick = useCallback(() => {
    if (!isValidating && !isVerifying) {
      fileInputRef.current?.click()
    }
  }, [isValidating, isVerifying])

  const isProcessing = isValidating || isVerifying

  return (
    <div className="w-full space-y-4 sm:space-y-6">
      {/* Upload Area */}
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-2xl sm:rounded-3xl p-6 sm:p-12 text-center transition-all duration-300 min-h-[280px] sm:min-h-[400px] flex flex-col items-center justify-center group ${
          isDragOver
            ? "border-green-400 bg-green-50 scale-[1.02] shadow-lg"
            : isProcessing
              ? "border-blue-400 bg-blue-50"
              : "border-gray-300 hover:border-green-400 hover:bg-green-50/50 cursor-pointer"
        } ${isProcessing ? "cursor-not-allowed" : ""}`}
      >
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-5 pointer-events-none">
          <div className="absolute top-4 sm:top-8 left-4 sm:left-8 w-2 sm:w-3 h-2 sm:h-3 bg-green-400 rounded-full animate-pulse"></div>
          <div
            className="absolute top-8 sm:top-16 right-6 sm:right-12 w-1.5 sm:w-2 h-1.5 sm:h-2 bg-emerald-400 rounded-full animate-pulse"
            style={{ animationDelay: "1s" }}
          ></div>
          <div
            className="absolute bottom-6 sm:bottom-12 left-8 sm:left-16 w-3 sm:w-4 h-3 sm:h-4 bg-teal-400 rounded-full animate-pulse"
            style={{ animationDelay: "2s" }}
          ></div>
          <div
            className="absolute bottom-4 sm:bottom-8 right-4 sm:right-8 w-1.5 sm:w-2 h-1.5 sm:h-2 bg-green-500 rounded-full animate-pulse"
            style={{ animationDelay: "3s" }}
          ></div>
        </div>

        <div className="space-y-4 sm:space-y-6 relative z-10 w-full max-w-md">
          {/* Icon */}
          <div className="relative mx-auto w-fit">
            <div
              className={`w-16 h-16 sm:w-20 sm:h-20 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 ${
                isDragOver
                  ? "bg-gradient-to-br from-green-400 to-emerald-500 scale-110 rotate-12"
                  : isProcessing
                    ? "bg-gradient-to-br from-blue-400 to-blue-500 animate-pulse"
                    : "bg-gradient-to-br from-gray-400 to-gray-500 group-hover:from-green-400 group-hover:to-emerald-500 group-hover:scale-110"
              }`}
            >
              {isVerifying ? (
                <Leaf className="w-8 h-8 sm:w-10 sm:h-10 text-white" />
              ) : isValidating ? (
                <FileImage className="w-8 h-8 sm:w-10 sm:h-10 text-white" />
              ) : (
                <Upload className="w-8 h-8 sm:w-10 sm:h-10 text-white" />
              )}
            </div>
            <div className="absolute -top-1 sm:-top-2 -right-1 sm:-right-2 w-5 h-5 sm:w-6 sm:h-6 bg-yellow-400 rounded-full flex items-center justify-center">
              <Sparkles className="w-2.5 h-2.5 sm:w-3 sm:h-3 text-white" />
            </div>
          </div>

          {/* Content */}
          <div className="space-y-3 sm:space-y-4">
            <h3 className="text-xl sm:text-2xl font-bold text-gray-900 leading-tight">
              {isDragOver
                ? "Drop your leaf image here!"
                : isVerifying
                  ? "Verifying leaf image..."
                  : isValidating
                    ? "Processing image..."
                    : "Upload Leaf Image"}
            </h3>
            <p className="text-gray-600 text-base sm:text-lg leading-relaxed px-2">
              {isDragOver
                ? "Release to start leaf verification"
                : isVerifying
                  ? "Checking if the uploaded image contains a leaf"
                  : isValidating
                    ? "Validating and preparing your image"
                    : "Drag and drop a leaf image here, or tap to browse"}
            </p>
          </div>

          {/* Progress Bar */}
          {(isValidating || isVerifying) && (
            <div className="w-full space-y-2">
              <Progress value={isVerifying ? 50 : uploadProgress} className="h-2" />
              <p className="text-sm text-gray-500">
                {isVerifying ? "Verifying leaf..." : `${uploadProgress}% complete`}
              </p>
            </div>
          )}

          {/* File Requirements */}
          {!isProcessing && (
            <div className="grid grid-cols-2 gap-2 sm:gap-4 text-sm">
              <div className="flex items-center gap-2 bg-white/80 backdrop-blur-sm px-3 sm:px-4 py-2 sm:py-3 rounded-lg sm:rounded-xl shadow-sm border border-gray-100">
                <ImageIcon className="w-3 h-3 sm:w-4 sm:h-4 text-green-600 flex-shrink-0" />
                <span className="text-gray-700 text-xs sm:text-sm">JPG, PNG, JPEG</span>
              </div>
              <div className="flex items-center gap-2 bg-white/80 backdrop-blur-sm px-3 sm:px-4 py-2 sm:py-3 rounded-lg sm:rounded-xl shadow-sm border border-gray-100">
                <Leaf className="w-3 h-3 sm:w-4 sm:h-4 text-green-600 flex-shrink-0" />
                <span className="text-gray-700 text-xs sm:text-sm">Leaf Images Only</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {validationError && (
        <Alert variant="destructive" className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="font-medium text-sm">{validationError}</AlertDescription>
        </Alert>
      )}

      {/* Mobile-optimized Professional Tips */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl sm:rounded-2xl p-4 sm:p-6 border border-blue-100">
        <div className="flex items-start gap-3 sm:gap-4">
          <div className="w-8 h-8 sm:w-10 sm:h-10 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
            <CheckCircle2 className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-blue-900 mb-2 sm:mb-3 text-sm sm:text-base">🍃 Leaf Photography Tips</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3 text-xs sm:text-sm text-blue-800">
              <div className="space-y-1.5 sm:space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>Capture individual leaves clearly</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>Focus on diseased areas</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>Use natural lighting</span>
                </div>
              </div>
              <div className="space-y-1.5 sm:space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>Avoid blurry images</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>Include both sides if needed</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 sm:w-1.5 sm:h-1.5 bg-blue-500 rounded-full flex-shrink-0"></div>
                  <span>Ensure leaf fills the frame</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alternative Upload Button */}
      {!isProcessing && (
        <div className="text-center">
          <Button
            onClick={handleClick}
            variant="outline"
            className="border-2 border-gray-200 hover:border-green-400 hover:bg-green-50 px-6 sm:px-8 py-2 sm:py-3 rounded-lg sm:rounded-xl font-medium transition-all duration-300 text-sm sm:text-base"
          >
            <Upload className="w-3 h-3 sm:w-4 sm:h-4 mr-2" />
            Browse Leaf Images
          </Button>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
        disabled={isProcessing}
      />
    </div>
  )
}
