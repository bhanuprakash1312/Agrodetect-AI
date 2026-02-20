"use client"

import type React from "react"
import { ArrowLeft, Microscope, Info, Download, Share2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip"
import { useToast } from "@/hooks/use-toast"

interface AnalysisSectionProps {
  selectedImage: File
  imagePreview: string
  isVerifyingLeaf: boolean
  isLoading: boolean
  onPredict: () => void
  onReset: () => void
  onDownloadReport: () => void
  onShare: () => void
  predictionResult: any
}

export const AnalysisSection: React.FC<AnalysisSectionProps> = ({
  selectedImage,
  imagePreview,
  isVerifyingLeaf,
  isLoading,
  onPredict,
  onReset,
  onDownloadReport,
  onShare,
  predictionResult,
}) => {
  const { toast } = useToast()

  return (
    <div className="space-y-4 sm:space-y-8">
      {/* Mobile-optimized Breadcrumb Navigation */}
      <nav className="flex items-center gap-2 text-sm text-gray-600 px-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onReset}
          className="text-gray-600 hover:text-gray-900 p-2 h-auto"
          disabled={isVerifyingLeaf || isLoading}
        >
          <ArrowLeft className="w-4 h-4 mr-1" />
          <span className="hidden sm:inline">Back to Upload</span>
          <span className="sm:hidden">Back</span>
        </Button>
        <span>/</span>
        <span className="text-gray-900 font-medium">Analysis</span>
        {predictionResult && (
          <>
            <span>/</span>
            <span className="text-gray-900 font-medium hidden sm:inline">Results</span>
            <span className="text-gray-900 font-medium sm:hidden">Done</span>
          </>
        )}
      </nav>

      {/* Mobile-optimized Image Preview */}
      <div className="bg-white/70 backdrop-blur-sm rounded-2xl sm:rounded-3xl shadow-xl border border-white/50 p-4 sm:p-8">
        <div className="flex flex-col gap-6 sm:gap-8">
          <div className="w-full">
            <div className="flex items-center justify-between mb-4 sm:mb-6">
              <div className="flex items-center gap-2 sm:gap-3">
                <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm">
                  1
                </div>
                <h3 className="text-lg sm:text-2xl font-bold text-gray-900">Image Preview</h3>
              </div>

              {/* Mobile-optimized image metadata */}
              <div className="text-right text-xs sm:text-sm text-gray-500">
                <p className="truncate max-w-24 sm:max-w-none">{selectedImage?.name}</p>
                <p>{selectedImage ? (selectedImage.size / 1024 / 1024).toFixed(2) : 0} MB</p>
              </div>
            </div>

            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-emerald-400/20 rounded-xl sm:rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative bg-gray-100 rounded-xl sm:rounded-2xl overflow-hidden border-2 sm:border-4 border-white shadow-lg">
                <img
                  src={imagePreview || "/placeholder.svg"}
                  alt="Selected crop"
                  className="w-full h-48 sm:h-64 lg:h-80 object-cover"
                />
                <div className="absolute top-2 sm:top-4 right-2 sm:right-4">
                  <Badge
                    className={`border-0 text-xs sm:text-sm ${
                      isVerifyingLeaf ? "bg-blue-100 text-blue-700" : "bg-white/90 text-gray-700"
                    }`}
                  >
                    {isVerifyingLeaf ? "Verifying Leaf..." : "Leaf Verified ✓"}
                  </Badge>
                </div>
              </div>
            </div>

            {/* Mobile-optimized image quality indicators */}
            <div className="grid grid-cols-3 gap-2 sm:gap-4 mt-4 sm:mt-6">
              <div className="text-center p-2 sm:p-3 bg-green-50 rounded-lg sm:rounded-xl border border-green-100">
                <div className="w-6 h-6 sm:w-8 sm:h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-1 sm:mb-2">
                  <span className="text-green-600 text-xs sm:text-sm">✓</span>
                </div>
                <p className="text-xs font-medium text-green-800">Leaf Detected</p>
              </div>
              <div className="text-center p-2 sm:p-3 bg-green-50 rounded-lg sm:rounded-xl border border-green-100">
                <div className="w-6 h-6 sm:w-8 sm:h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-1 sm:mb-2">
                  <span className="text-green-600 text-xs sm:text-sm">✓</span>
                </div>
                <p className="text-xs font-medium text-green-800">Good Quality</p>
              </div>
              <div className="text-center p-2 sm:p-3 bg-green-50 rounded-lg sm:rounded-xl border border-green-100">
                <div className="w-6 h-6 sm:w-8 sm:h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-1 sm:mb-2">
                  <span className="text-green-600 text-xs sm:text-sm">✓</span>
                </div>
                <p className="text-xs font-medium text-green-800">Ready to Analyze</p>
              </div>
            </div>
          </div>

          {/* Mobile-optimized Action Panel */}
          <div className="space-y-4 sm:space-y-6">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-r from-green-500 to-green-600 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm">
                2
              </div>
              <h4 className="text-lg sm:text-xl font-bold text-gray-900">Analysis Control</h4>
            </div>

            {/* Mobile-optimized settings */}
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl sm:rounded-2xl p-4 sm:p-6 border border-gray-200">
              <h5 className="font-semibold text-gray-900 mb-3 text-sm sm:text-base">Analysis Settings</h5>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3 text-xs sm:text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Model:</span>
                  <span className="font-medium text-gray-900">CNN v2.1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Confidence:</span>
                  <span className="font-medium text-gray-900">85%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Leaf Status:</span>
                  <span className="font-medium text-green-600">Verified ✓</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Mode:</span>
                  <span className="font-medium text-gray-900">High Accuracy</span>
                </div>
              </div>
            </div>

            {/* Mobile-optimized action buttons */}
            <div className="space-y-3 sm:space-y-4">
              <Button
                onClick={onPredict}
                disabled={isLoading || isVerifyingLeaf}
                className="w-full h-12 sm:h-14 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold rounded-xl sm:rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="flex items-center justify-center gap-2 sm:gap-3">
                  <Microscope className="w-4 h-4 sm:w-5 sm:h-5" />
                  <span className="text-sm sm:text-base">
                    {isLoading ? "Analyzing..." : isVerifyingLeaf ? "Verifying..." : "Start AI Analysis"}
                  </span>
                </div>
              </Button>

              <Button
                onClick={onReset}
                variant="outline"
                disabled={isVerifyingLeaf || isLoading}
                className="w-full h-10 sm:h-12 border-2 border-gray-200 hover:border-gray-300 rounded-xl sm:rounded-2xl font-medium text-sm sm:text-base"
              >
                Select Different Image
              </Button>
            </div>

            {/* Mobile-optimized analysis info */}
            <div className="bg-blue-50 rounded-xl sm:rounded-2xl p-3 sm:p-4 border border-blue-100">
              <h6 className="font-medium text-blue-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                <Info className="w-3 h-3 sm:w-4 sm:h-4" />
                Analysis Information
              </h6>
              <ul className="text-xs sm:text-sm text-blue-800 space-y-1">
                <li>• Leaf verification: Complete ✓</li>
                <li>• Analysis time: 3-5 seconds</li>
                <li>• Supports 50+ crop diseases</li>
                <li>• Includes treatment recommendations</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Results Action Buttons */}
      {predictionResult && !isLoading && (
        <div className="flex items-center justify-between mb-4 sm:mb-6">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm">
              4
            </div>
            <h3 className="text-lg sm:text-2xl font-bold text-gray-900">Analysis Results</h3>
          </div>

          {/* Mobile-optimized action buttons */}
          <div className="flex gap-1 sm:gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={onDownloadReport}
                    variant="outline"
                    size="sm"
                    className="border-gray-200 hover:border-gray-300 p-2 h-8 w-8 sm:h-9 sm:w-9"
                  >
                    <Download className="w-3 h-3 sm:w-4 sm:h-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Download Report</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    onClick={onShare}
                    variant="outline"
                    size="sm"
                    className="border-gray-200 hover:border-gray-300 p-2 h-8 w-8 sm:h-9 sm:w-9"
                  >
                    <Share2 className="w-3 h-3 sm:w-4 sm:h-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Share Results</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        </div>
      )}
    </div>
  )
}
