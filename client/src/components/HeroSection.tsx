"use client"

import type React from "react"

import { Microscope, Shield, TrendingUp } from "lucide-react"

export const HeroSection: React.FC = () => {
  return (
    <div className="text-center mb-8 sm:mb-12">
      <div className="inline-flex items-center gap-2 bg-green-100 text-green-800 px-3 sm:px-4 py-2 rounded-full text-xs sm:text-sm font-medium mb-4 sm:mb-6">
        <Microscope className="w-3 h-3 sm:w-4 sm:h-4 text-green-600 flex-shrink-0" />
        AI-Powered Disease Detection
      </div>

      <h1 className="text-2xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-3 sm:mb-4 tracking-tight leading-tight">
        Professional Crop
        <br className="sm:hidden" />
        <span className="bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
          {" "}
          Health Analysis
        </span>
      </h1>
      <p className="text-base sm:text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed px-2">
        Upload high-quality images of your crops to receive instant AI-powered disease detection and evidence-based
        treatment recommendations
      </p>

      {/* Mobile-optimized trust indicators */}
      <div className="flex flex-col sm:flex-row flex-wrap justify-center gap-3 sm:gap-6 mt-6 sm:mt-8 px-4">
        <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
          <Shield className="w-4 h-4 text-green-600 flex-shrink-0" />
          <span>99.2% Accuracy</span>
        </div>
        <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
          <TrendingUp className="w-4 h-4 text-green-600 flex-shrink-0" />
          <span>50,000+ Analyses</span>
        </div>
        <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
          <Microscope className="w-4 h-4 text-green-600 flex-shrink-0" />
          <span>Real-time Processing</span>
        </div>
      </div>
    </div>
  )
}
