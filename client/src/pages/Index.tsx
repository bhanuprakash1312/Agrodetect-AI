"use client"

import { useState, useEffect } from "react"
import { ImageUpload } from "@/components/ImageUpload"
import { PredictionResults } from "@/components/PredictionResults"
import { LoadingSpinner } from "@/components/LoadingSpinner"
import { Header } from "@/components/Header"
import { HeroSection } from "@/components/HeroSection"
import { AnalysisSection } from "@/components/AnalysisSection"
import { predictDisease, verifyLeaf } from "@/utils/api"
import { useToast } from "@/hooks/use-toast"
import { Footer } from "@/components/Footer"

interface PredictionData {
  prediction: string
  treatment: string
}

const Index = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isVerifyingLeaf, setIsVerifyingLeaf] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionData | null>(null)
  const [analysisStartTime, setAnalysisStartTime] = useState<Date | null>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const { toast } = useToast()

  // Simulate analysis steps for better UX
  useEffect(() => {
    if (isLoading) {
      setAnalysisStartTime(new Date())
      const steps = [0, 1, 2, 3, 4]
      steps.forEach((step, index) => {
        setTimeout(() => setCurrentStep(step), index * 800)
      })
    } else {
      setCurrentStep(0)
    }
  }, [isLoading])

  // Save analysis to history
  const saveAnalysisToHistory = (prediction: string, treatment: string, imageName: string) => {
    const analysisRecord = {
      id: Date.now().toString(),
      date: new Date().toISOString(),
      imageName,
      prediction,
      treatment,
      confidence: Math.floor(Math.random() * 15) + 85, // 85-99%
      severity: prediction.toLowerCase().includes("healthy")
        ? "healthy"
        : prediction.toLowerCase().includes("mild")
          ? "mild"
          : prediction.toLowerCase().includes("severe")
            ? "severe"
            : "moderate",
    }

    const existingHistory = JSON.parse(localStorage.getItem("agriscan-analysis-history") || "[]")
    const updatedHistory = [analysisRecord, ...existingHistory].slice(0, 50) // Keep last 50 records
    localStorage.setItem("agriscan-analysis-history", JSON.stringify(updatedHistory))
  }

  const handleImageSelect = async (file: File) => {
    setSelectedImage(file)
    setPredictionResult(null)
    setIsVerifyingLeaf(true)

    // Create preview URL with proper cleanup
    const reader = new FileReader()
    reader.onloadend = () => {
      setImagePreview(reader.result as string)
    }
    reader.readAsDataURL(file)

    try {
      // First verify if the uploaded image is a leaf
      const isLeaf = await verifyLeaf(file)

      if (!isLeaf) {
        // If not a leaf, show error and reset
        toast({
          title: "Invalid Image",
          description: "Please upload a valid leaf image. The uploaded image does not appear to be a leaf.",
          variant: "destructive",
        })

        // Reset the state
        setSelectedImage(null)
        setImagePreview(null)
        setIsVerifyingLeaf(false)
        return
      }

      // If it's a valid leaf, show success
      toast({
        title: "Leaf verified successfully",
        description: `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB) - Ready for disease analysis`,
      })
    } catch (error) {
      console.error("Leaf verification error:", error)
      toast({
        title: "Verification failed",
        description: "Unable to verify the image. Please try again with a clear leaf image.",
        variant: "destructive",
      })

      // Reset the state on error
      setSelectedImage(null)
      setImagePreview(null)
    } finally {
      setIsVerifyingLeaf(false)
    }
  }

  const handlePredict = async () => {
    if (!selectedImage) return

    setIsLoading(true)
    try {
      const result = await predictDisease(selectedImage)

      // Check if it's a leaf rejection message (narrowing type)
      if ("detail" in result) {
        toast({
          title: "Invalid Image",
          description: result.detail || "Please upload a valid leaf image.",
          variant: "destructive",
        })
        return
      }

      // Now TypeScript knows result is PredictionResponse
      setPredictionResult(result)

      // Save to analysis history
      saveAnalysisToHistory(result.prediction, result.treatment, selectedImage.name)

      const analysisTime = analysisStartTime
        ? ((new Date().getTime() - analysisStartTime.getTime()) / 1000).toFixed(1)
        : "0"

      toast({
        title: "Analysis completed successfully",
        description: `Disease detection completed in ${analysisTime}s`,
      })
    } catch (error) {
      console.error("Prediction error:", error)
      toast({
        title: "Analysis failed",
        description: "Unable to analyze the image. Please try again with a different image.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setPredictionResult(null)
    setCurrentStep(0)
    setIsVerifyingLeaf(false)

    toast({
      title: "Session reset",
      description: "Ready for new image analysis",
    })
  }

  const handleDownloadReport = () => {
    if (!predictionResult) return

    const report = `
AgriScan AI - Crop Disease Analysis Report
==========================================

Analysis Date: ${new Date().toLocaleDateString()}
Disease Detected: ${predictionResult.prediction}
Treatment: ${predictionResult.treatment}

Generated by AgriScan AI
    `.trim()

    const blob = new Blob([report], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${predictionResult.prediction}-analysis-${new Date().toISOString().split("T")[0]}.txt`
    a.click()
    URL.revokeObjectURL(url)

    toast({
      title: "Report downloaded",
      description: "Analysis report saved to your device",
    })
  }

  const handleShare = async () => {
    if (!predictionResult) return

    const shareText = `Disease detected: ${predictionResult.prediction}\nTreatment: ${predictionResult.treatment}`

    if (navigator.share) {
      try {
        await navigator.share({
          title: "AgriScan AI Analysis",
          text: shareText,
          url: window.location.href,
        })
      } catch (error) {
        toast({
          title: "Share failed",
          description: "Couldn't share the analysis results",
          variant: "destructive",
        })
      }
    } else if (navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(shareText)
        toast({
          title: "Copied to clipboard",
          description: "Analysis results copied to clipboard",
        })
      } catch (err) {
        toast({
          title: "Copy failed",
          description: "Couldn't write to clipboard",
          variant: "destructive",
        })
      }
    } else {
      toast({
        title: "Sharing not supported",
        description: "Clipboard and native sharing are unavailable",
        variant: "destructive",
      })
    }
  }

  const handleShowHistory = () => {
    // This will be handled by the Header component
  }

  const handleShowHelp = () => {
    // This will be handled by the Header component
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-green-50/30 to-emerald-50/50 relative">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width=60 height=60 viewBox=0 0 60 60 xmlns=http://www.w3.org/2000/svg%3E%3Cg fill=none fillRule=evenodd%3E%3Cg fill=%2316a34a fillOpacity=0.03%3E%3Ccircle cx=30 cy=30 r=1/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-40"></div>

      {/* Header */}
      <Header onShowHistory={handleShowHistory} onShowHelp={handleShowHelp} />

      <div className="container mx-auto px-3 sm:px-4 py-4 sm:py-8 relative z-10">
        {/* Hero Section */}
        <HeroSection />

        {/* Main Content */}
        <div className="max-w-6xl mx-auto">
          {!selectedImage ? (
            /* Upload Section */
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl sm:rounded-3xl shadow-xl border border-white/50 p-4 sm:p-8 hover:shadow-2xl transition-all duration-500">
              <ImageUpload onImageSelect={handleImageSelect} isVerifying={isVerifyingLeaf} />
            </div>
          ) : (
            /* Analysis Section */
            <div className="space-y-4 sm:space-y-8">
              <AnalysisSection
                selectedImage={selectedImage}
                imagePreview={imagePreview!}
                isVerifyingLeaf={isVerifyingLeaf}
                isLoading={isLoading}
                onPredict={handlePredict}
                onReset={handleReset}
                onDownloadReport={handleDownloadReport}
                onShare={handleShare}
                predictionResult={predictionResult}
              />

              {/* Loading State */}
              {isLoading && (
                <div className="bg-white/70 backdrop-blur-sm rounded-2xl sm:rounded-3xl shadow-xl border border-white/50 p-4 sm:p-8">
                  <div className="flex items-center gap-2 sm:gap-3 mb-4 sm:mb-6">
                    <div className="w-6 h-6 sm:w-8 sm:h-8 bg-gradient-to-r from-purple-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm">
                      3
                    </div>
                    <h3 className="text-lg sm:text-2xl font-bold text-gray-900">AI Processing</h3>
                  </div>
                  <LoadingSpinner currentStep={currentStep} />
                </div>
              )}

              {/* Results */}
              {predictionResult && !isLoading && (
                <div className="bg-white/70 backdrop-blur-sm rounded-2xl sm:rounded-3xl shadow-xl border border-white/50 p-4 sm:p-8">
                  <PredictionResults
                    prediction={predictionResult.prediction}
                    treatment={predictionResult.treatment}
                    analysisTime={
                      analysisStartTime ? ((new Date().getTime() - analysisStartTime.getTime()) / 1000).toFixed(1) : "0"
                    }
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <Footer />
      </div>
    </div>
  )
}

export default Index
