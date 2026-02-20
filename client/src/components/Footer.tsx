"use client"

import type React from "react"

export const Footer: React.FC = () => {
  return (
    <footer className="text-center mt-12 sm:mt-16 pb-6 sm:pb-8">
      <div className="bg-white/60 backdrop-blur-sm rounded-xl sm:rounded-2xl p-4 sm:p-6 max-w-2xl mx-auto border border-white/30">
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-6 mb-3 sm:mb-4">
          <div className="flex items-center gap-2 text-xs sm:text-sm text-gray-600">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span>Secure Processing</span>
          </div>
          <div className="flex items-center gap-2 text-xs sm:text-sm text-gray-600">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span>GDPR Compliant</span>
          </div>
          <div className="flex items-center gap-2 text-xs sm:text-sm text-gray-600">
            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
            <span>ISO 27001 Certified</span>
          </div>
        </div>
        <p className="text-gray-600 font-medium mb-2 text-sm sm:text-base">Powered by Advanced Machine Learning</p>
        <p className="text-xs sm:text-sm text-gray-500 leading-relaxed">
          Supporting agricultural professionals worldwide with intelligent crop health monitoring
        </p>
      </div>
    </footer>
  )
}
