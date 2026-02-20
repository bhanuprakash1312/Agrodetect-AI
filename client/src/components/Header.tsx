"use client"

import type React from "react"

import { useState } from "react"
import { Leaf, History, Info, Menu } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { AnalysisHistory } from "./AnalysisHistory"
import { HelpDocumentation } from "./HelpDocumentation"

interface HeaderProps {
  onShowHistory: () => void
  onShowHelp: () => void
}

export const Header: React.FC<HeaderProps> = ({ onShowHistory, onShowHelp }) => {
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [isHelpOpen, setIsHelpOpen] = useState(false)

  const handleShowHistory = () => {
    setIsHistoryOpen(true)
    onShowHistory()
  }

  const handleShowHelp = () => {
    setIsHelpOpen(true)
    onShowHelp()
  }

  // Mobile menu component
  const MobileMenu = () => (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="md:hidden">
          <Menu className="w-5 h-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-80">
        <div className="flex flex-col gap-4 mt-6">
          <Button variant="ghost" className="justify-start" onClick={handleShowHistory}>
            <History className="w-4 h-4 mr-2" />
            Analysis History
          </Button>
          <Button variant="ghost" className="justify-start" onClick={handleShowHelp}>
            <Info className="w-4 h-4 mr-2" />
            Help & Documentation
          </Button>
          <div className="pt-4 border-t">
            <Badge variant="outline" className="text-green-700 border-green-200 w-full justify-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
              System Online
            </Badge>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )

  return (
    <>
      <header className="z-10 border-b border-white/20 bg-white/80 backdrop-blur-md sticky top-0">
        <div className="container mx-auto px-3 sm:px-4 py-3 sm:py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg">
                <Leaf className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
              </div>
              <div>
                <h1 className="text-lg sm:text-xl font-bold text-gray-900">AgriScan AI</h1>
                <p className="text-xs text-gray-500 hidden sm:block">Professional Crop Analysis</p>
              </div>
            </div>

            {/* Desktop menu */}
            <div className="hidden md:flex items-center gap-2">
              <Badge variant="outline" className="text-green-700 border-green-200">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                Online
              </Badge>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="sm" onClick={handleShowHistory}>
                      <History className="w-4 h-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Analysis History</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="sm" onClick={handleShowHelp}>
                      <Info className="w-4 h-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Help & Documentation</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>

            {/* Mobile menu */}
            <MobileMenu />
          </div>
        </div>
      </header>

      {/* Analysis History Modal */}
      <AnalysisHistory isOpen={isHistoryOpen} onClose={() => setIsHistoryOpen(false)} />

      {/* Help Documentation Modal */}
      <HelpDocumentation isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </>
  )
}
