"use client"

import type React from "react"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  HelpCircle,
  Camera,
  Microscope,
  Smartphone,
  Leaf,
  AlertTriangle,
  CheckCircle,
  Info,
  BookOpen,
  Settings,
  Phone,
  X,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"

interface HelpDocumentationProps {
  isOpen: boolean
  onClose: () => void
}

export const HelpDocumentation: React.FC<HelpDocumentationProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState("getting-started")

  const tabs = [
    { id: "getting-started", label: "Getting Started", shortLabel: "Start", icon: BookOpen },
    { id: "photography", label: "Photography", shortLabel: "Photo", icon: Camera },
    { id: "analysis", label: "Analysis", shortLabel: "AI", icon: Microscope },
    { id: "troubleshooting", label: "Support", shortLabel: "Help", icon: Settings },
  ]

  const currentTabIndex = tabs.findIndex((tab) => tab.id === activeTab)

  const goToPrevTab = () => {
    if (currentTabIndex > 0) {
      setActiveTab(tabs[currentTabIndex - 1].id)
    }
  }

  const goToNextTab = () => {
    if (currentTabIndex < tabs.length - 1) {
      setActiveTab(tabs[currentTabIndex + 1].id)
    }
  }

  const renderTabIcon = (tab) => {
    const IconComponent = tab.icon
    return <IconComponent className="w-4 h-4 text-blue-600" />
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="w-[95vw] max-w-4xl h-[90vh] max-h-[90vh] p-0 gap-0">
        {/* Mobile-optimized Header */}
        <DialogHeader className="p-4 sm:p-6 pb-0 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <DialogTitle className="text-lg sm:text-2xl font-bold flex items-center gap-2 sm:gap-3">
              <HelpCircle className="w-5 h-5 sm:w-6 sm:h-6 text-blue-600" />
              <span className="hidden sm:inline">Help & Documentation</span>
              <span className="sm:hidden">Help</span>
            </DialogTitle>
            <Button variant="ghost" size="sm" onClick={onClose} className="md:hidden p-2 h-8 w-8">
              <X className="w-4 h-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-hidden">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
            {/* Mobile-optimized Tabs */}
            <div className="border-b border-gray-200">
              {/* Desktop Tabs */}
              <TabsList className="hidden sm:grid w-full grid-cols-4 h-12 bg-gray-50">
                {tabs.map((tab) => (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="flex items-center gap-2 text-sm data-[state=active]:bg-white"
                  >
                    {renderTabIcon(tab)}
                    {tab.label}
                  </TabsTrigger>
                ))}
              </TabsList>

              {/* Mobile Tabs with Navigation */}
              <div className="sm:hidden bg-gray-50">
                <div className="flex items-center justify-between p-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={goToPrevTab}
                    disabled={currentTabIndex === 0}
                    className="p-2 h-8 w-8"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>

                  <div className="flex-1 text-center">
                    <div className="flex items-center justify-center gap-2">
                      {tabs[currentTabIndex] && (
                        <>
                          {renderTabIcon(tabs[currentTabIndex])}
                          <span className="font-medium text-sm">{tabs[currentTabIndex].shortLabel}</span>
                        </>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {currentTabIndex + 1} of {tabs.length}
                    </div>
                  </div>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={goToNextTab}
                    disabled={currentTabIndex === tabs.length - 1}
                    className="p-2 h-8 w-8"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                </div>

                {/* Mobile Tab Indicators */}
                <div className="flex justify-center gap-1 pb-2">
                  {tabs.map((_, index) => (
                    <div
                      key={index}
                      className={`h-1 w-8 rounded-full transition-colors ${
                        index === currentTabIndex ? "bg-blue-600" : "bg-gray-300"
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden">
              <ScrollArea className="h-full">
                <div className="p-4 sm:p-6">
                  <TabsContent value="getting-started" className="space-y-4 sm:space-y-6 mt-0">
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-3 sm:p-4 border border-green-200">
                      <h3 className="font-semibold text-green-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                        <BookOpen className="w-4 h-4 sm:w-5 sm:h-5" />
                        Welcome to AgriScan AI
                      </h3>
                      <p className="text-green-800 text-xs sm:text-sm leading-relaxed">
                        AgriScan AI is a professional crop disease detection system powered by advanced machine
                        learning. Follow this guide to get the most accurate results.
                      </p>
                    </div>

                    <Accordion type="single" collapsible className="w-full">
                      <AccordionItem value="step-1">
                        <AccordionTrigger className="text-left text-sm sm:text-base">
                          <div className="flex items-center gap-2 sm:gap-3">
                            <div className="w-5 h-5 sm:w-6 sm:h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-xs sm:text-sm font-bold flex-shrink-0">
                              1
                            </div>
                            How to Upload Images
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <p className="text-gray-700">
                            Upload high-quality images of individual leaves showing potential disease symptoms:
                          </p>
                          <ul className="list-disc list-inside space-y-1 text-xs sm:text-sm text-gray-600 ml-4">
                            <li>Click the upload area or drag and drop your image</li>
                            <li>Supported formats: JPG, PNG, JPEG, WebP</li>
                            <li>Maximum file size: 10MB</li>
                            <li>Minimum resolution: 300x300 pixels recommended</li>
                          </ul>
                          <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                            <p className="text-blue-800 text-xs sm:text-sm">
                              <strong>Pro Tip:</strong> The system automatically verifies that your image contains a
                              leaf before analysis.
                            </p>
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="step-2">
                        <AccordionTrigger className="text-left text-sm sm:text-base">
                          <div className="flex items-center gap-2 sm:gap-3">
                            <div className="w-5 h-5 sm:w-6 sm:h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-xs sm:text-sm font-bold flex-shrink-0">
                              2
                            </div>
                            Understanding Results
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <p className="text-gray-700">Your analysis results include:</p>
                          <div className="grid grid-cols-1 gap-3">
                            <div className="bg-gray-50 p-3 rounded-lg">
                              <h4 className="font-medium text-gray-900 mb-1 text-sm sm:text-base">Disease Detection</h4>
                              <p className="text-xs sm:text-sm text-gray-600">
                                Specific disease identification with confidence level
                              </p>
                            </div>
                            <div className="bg-gray-50 p-3 rounded-lg">
                              <h4 className="font-medium text-gray-900 mb-1 text-sm sm:text-base">Treatment Plan</h4>
                              <p className="text-xs sm:text-sm text-gray-600">
                                Evidence-based treatment recommendations
                              </p>
                            </div>
                            <div className="bg-gray-50 p-3 rounded-lg">
                              <h4 className="font-medium text-gray-900 mb-1 text-sm sm:text-base">
                                Severity Assessment
                              </h4>
                              <p className="text-xs sm:text-sm text-gray-600">Disease progression and urgency level</p>
                            </div>
                            <div className="bg-gray-50 p-3 rounded-lg">
                              <h4 className="font-medium text-gray-900 mb-1 text-sm sm:text-base">Prevention Tips</h4>
                              <p className="text-xs sm:text-sm text-gray-600">Future prevention strategies</p>
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="step-3">
                        <AccordionTrigger className="text-left text-sm sm:text-base">
                          <div className="flex items-center gap-2 sm:gap-3">
                            <div className="w-5 h-5 sm:w-6 sm:h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-xs sm:text-sm font-bold flex-shrink-0">
                              3
                            </div>
                            Supported Crops & Diseases
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <p className="text-gray-700">
                            AgriScan AI supports 50+ diseases across major crop categories:
                          </p>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div>
                              <h4 className="font-medium text-gray-900 mb-2 text-sm sm:text-base">Supported Crops</h4>
                              <ul className="text-xs sm:text-sm text-gray-600 space-y-1">
                                <li>• Tomato</li>
                                <li>• Potato</li>
                                <li>• Corn (Maize)</li>
                                <li>• Apple</li>
                                <li>• Grape</li>
                                <li>• Pepper</li>
                                <li>• And many more...</li>
                              </ul>
                            </div>
                            <div>
                              <h4 className="font-medium text-gray-900 mb-2 text-sm sm:text-base">Disease Types</h4>
                              <ul className="text-xs sm:text-sm text-gray-600 space-y-1">
                                <li>• Bacterial infections</li>
                                <li>• Fungal diseases</li>
                                <li>• Viral infections</li>
                                <li>• Nutrient deficiencies</li>
                                <li>• Pest damage</li>
                                <li>• Environmental stress</li>
                              </ul>
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  </TabsContent>

                  <TabsContent value="photography" className="space-y-4 sm:space-y-6 mt-0">
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-3 sm:p-4 border border-blue-200">
                      <h3 className="font-semibold text-blue-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                        <Camera className="w-4 h-4 sm:w-5 sm:h-5" />
                        Photography Best Practices
                      </h3>
                      <p className="text-blue-800 text-xs sm:text-sm leading-relaxed">
                        High-quality images are crucial for accurate disease detection. Follow these guidelines for
                        optimal results.
                      </p>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-green-50 rounded-lg p-3 sm:p-4 border border-green-200">
                        <h4 className="font-semibold text-green-900 mb-3 flex items-center gap-2 text-sm sm:text-base">
                          <CheckCircle className="w-4 h-4" />
                          Do's
                        </h4>
                        <ul className="space-y-2 text-xs sm:text-sm text-green-800">
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-green-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Use natural daylight for photography</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-green-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Focus on individual leaves with symptoms</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-green-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Fill the frame with the leaf</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-green-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Capture both sides if symptoms vary</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-green-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Ensure sharp focus on disease areas</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-green-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Use a plain background when possible</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-red-50 rounded-lg p-3 sm:p-4 border border-red-200">
                        <h4 className="font-semibold text-red-900 mb-3 flex items-center gap-2 text-sm sm:text-base">
                          <AlertTriangle className="w-4 h-4" />
                          Don'ts
                        </h4>
                        <ul className="space-y-2 text-xs sm:text-sm text-red-800">
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-red-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Avoid blurry or out-of-focus images</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-red-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Don't use flash or artificial lighting</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-red-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Avoid images with multiple leaves</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-red-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Don't photograph in low light conditions</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-red-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Avoid heavily shadowed areas</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <div className="w-1.5 h-1.5 bg-red-600 rounded-full mt-1.5 flex-shrink-0"></div>
                            <span>Don't include non-plant objects</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-yellow-50 rounded-lg p-3 sm:p-4 border border-yellow-200">
                      <h4 className="font-semibold text-yellow-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                        <Smartphone className="w-4 h-4" />
                        Mobile Photography Tips
                      </h4>
                      <div className="grid grid-cols-1 gap-3 text-xs sm:text-sm text-yellow-800">
                        <div>
                          <p className="font-medium mb-1">Camera Settings:</p>
                          <ul className="space-y-1">
                            <li>• Use the highest resolution available</li>
                            <li>• Enable HDR mode for better detail</li>
                            <li>• Tap to focus on the disease area</li>
                          </ul>
                        </div>
                        <div>
                          <p className="font-medium mb-1">Positioning:</p>
                          <ul className="space-y-1">
                            <li>• Hold the phone steady</li>
                            <li>• Get close but maintain focus</li>
                            <li>• Avoid camera shake</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="analysis" className="space-y-4 sm:space-y-6 mt-0">
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-3 sm:p-4 border border-purple-200">
                      <h3 className="font-semibold text-purple-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                        <Microscope className="w-4 h-4 sm:w-5 sm:h-5" />
                        Understanding AI Analysis
                      </h3>
                      <p className="text-purple-800 text-xs sm:text-sm leading-relaxed">
                        Learn how our AI system works and how to interpret the results for maximum benefit.
                      </p>
                    </div>

                    <Accordion type="single" collapsible className="w-full">
                      <AccordionItem value="how-it-works">
                        <AccordionTrigger className="text-sm sm:text-base">How the AI Analysis Works</AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <p className="text-gray-700">Our AI system uses a multi-step process:</p>
                          <div className="space-y-3">
                            {[
                              {
                                step: 1,
                                title: "Image Preprocessing",
                                description: "Enhances image quality and normalizes lighting conditions",
                              },
                              {
                                step: 2,
                                title: "Feature Extraction",
                                description: "Identifies key visual patterns and disease markers",
                              },
                              {
                                step: 3,
                                title: "Disease Classification",
                                description: "Compares patterns against trained disease database",
                              },
                              {
                                step: 4,
                                title: "Treatment Recommendation",
                                description: "Generates evidence-based treatment protocols",
                              },
                            ].map((item) => (
                              <div key={item.step} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                                <div className="w-5 h-5 sm:w-6 sm:h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-xs sm:text-sm font-bold flex-shrink-0">
                                  {item.step}
                                </div>
                                <div className="flex-1">
                                  <h4 className="font-medium text-gray-900 text-sm sm:text-base">{item.title}</h4>
                                  <p className="text-xs sm:text-sm text-gray-600">{item.description}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="confidence-levels">
                        <AccordionTrigger className="text-sm sm:text-base">
                          Understanding Confidence Levels
                        </AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <p className="text-gray-700">
                            Confidence levels indicate how certain the AI is about its diagnosis:
                          </p>
                          <div className="space-y-2">
                            {[
                              {
                                range: "95-99%",
                                level: "Very High",
                                description: "Extremely reliable diagnosis",
                                color: "green",
                              },
                              {
                                range: "90-94%",
                                level: "High",
                                description: "Reliable diagnosis, proceed with confidence",
                                color: "blue",
                              },
                              {
                                range: "85-89%",
                                level: "Good",
                                description: "Consider additional verification",
                                color: "yellow",
                              },
                              {
                                range: "80-84%",
                                level: "Moderate",
                                description: "Seek expert consultation",
                                color: "orange",
                              },
                            ].map((item) => (
                              <div
                                key={item.range}
                                className={`flex items-center gap-3 p-2 bg-${item.color}-50 rounded border border-${item.color}-200`}
                              >
                                <Badge className={`bg-${item.color}-100 text-${item.color}-800 text-xs`}>
                                  {item.range}
                                </Badge>
                                <span className={`text-xs sm:text-sm text-${item.color}-800 flex-1`}>
                                  <strong>{item.level}</strong> - {item.description}
                                </span>
                              </div>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="severity-levels">
                        <AccordionTrigger className="text-sm sm:text-base">Severity Assessment Guide</AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {[
                              {
                                level: "Healthy",
                                description: "No disease detected. Continue regular monitoring.",
                                color: "green",
                              },
                              {
                                level: "Mild",
                                description: "Early stage infection. Immediate treatment recommended.",
                                color: "yellow",
                              },
                              {
                                level: "Moderate",
                                description: "Established infection. Aggressive treatment needed.",
                                color: "orange",
                              },
                              {
                                level: "Severe",
                                description: "Advanced infection. Urgent intervention required.",
                                color: "red",
                              },
                            ].map((item) => (
                              <div
                                key={item.level}
                                className={`p-3 bg-${item.color}-50 rounded-lg border border-${item.color}-200`}
                              >
                                <h4 className={`font-medium text-${item.color}-900 mb-2 text-sm sm:text-base`}>
                                  {item.level}
                                </h4>
                                <p className={`text-xs sm:text-sm text-${item.color}-800`}>{item.description}</p>
                              </div>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  </TabsContent>

                  <TabsContent value="troubleshooting" className="space-y-4 sm:space-y-6 mt-0">
                    <div className="bg-gradient-to-r from-orange-50 to-red-50 rounded-lg p-3 sm:p-4 border border-orange-200">
                      <h3 className="font-semibold text-orange-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                        <Settings className="w-4 h-4 sm:w-5 sm:h-5" />
                        Troubleshooting & Support
                      </h3>
                      <p className="text-orange-800 text-xs sm:text-sm leading-relaxed">
                        Common issues and solutions to help you get the most out of AgriScan AI.
                      </p>
                    </div>

                    <Accordion type="single" collapsible className="w-full">
                      <AccordionItem value="upload-issues">
                        <AccordionTrigger className="text-sm sm:text-base">Image Upload Problems</AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <div className="space-y-3">
                            {[
                              {
                                title: "File too large error",
                                solution: "Compress your image or use a different format. Maximum size is 10MB.",
                              },
                              {
                                title: "Unsupported file format",
                                solution: "Use JPG, PNG, JPEG, or WebP formats only.",
                              },
                              {
                                title: "Upload fails repeatedly",
                                solution: "Check your internet connection and try refreshing the page.",
                              },
                            ].map((item) => (
                              <div key={item.title} className="p-3 bg-gray-50 rounded-lg">
                                <h4 className="font-medium text-gray-900 mb-1 text-sm sm:text-base">{item.title}</h4>
                                <p className="text-xs sm:text-sm text-gray-600">{item.solution}</p>
                              </div>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="analysis-issues">
                        <AccordionTrigger className="text-sm sm:text-base">Analysis Problems</AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <div className="space-y-3">
                            {[
                              {
                                title: '"Not a leaf" error',
                                solution:
                                  "Ensure your image clearly shows a single leaf. Avoid images with multiple objects or unclear subjects.",
                              },
                              {
                                title: "Low confidence results",
                                solution:
                                  "Try uploading a clearer image with better lighting and focus on the diseased area.",
                              },
                              {
                                title: "Analysis takes too long",
                                solution: "Large images may take longer to process. Try using a smaller file size.",
                              },
                            ].map((item) => (
                              <div key={item.title} className="p-3 bg-gray-50 rounded-lg">
                                <h4 className="font-medium text-gray-900 mb-1 text-sm sm:text-base">{item.title}</h4>
                                <p className="text-xs sm:text-sm text-gray-600">{item.solution}</p>
                              </div>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="contact-support">
                        <AccordionTrigger className="text-sm sm:text-base">Contact Support</AccordionTrigger>
                        <AccordionContent className="space-y-3 text-sm sm:text-base">
                          <div className="grid grid-cols-1 gap-4">
                            <div className="p-3 sm:p-4 bg-green-50 rounded-lg border border-green-200">
                              <h4 className="font-medium text-green-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                                <Phone className="w-4 h-4" />
                                Technical Support
                              </h4>
                              <p className="text-xs sm:text-sm text-green-800 mb-2">
                                For technical issues and system problems:
                              </p>
                              <ul className="text-xs sm:text-sm text-green-800 space-y-1">
                                <li>• Email: support@agriscan.ai</li>
                                <li>• Response time: 24-48 hours</li>
                                <li>• Include error screenshots</li>
                              </ul>
                            </div>
                            <div className="p-3 sm:p-4 bg-blue-50 rounded-lg border border-blue-200">
                              <h4 className="font-medium text-blue-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                                <Leaf className="w-4 h-4" />
                                Agricultural Experts
                              </h4>
                              <p className="text-xs sm:text-sm text-blue-800 mb-2">For complex disease questions:</p>
                              <ul className="text-xs sm:text-sm text-blue-800 space-y-1">
                                <li>• Email: experts@agriscan.ai</li>
                                <li>• Consultation fee may apply</li>
                                <li>• Include analysis results</li>
                              </ul>
                            </div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>

                    <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-3 sm:p-4 border border-gray-200">
                      <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2 text-sm sm:text-base">
                        <Info className="w-4 h-4" />
                        System Information
                      </h4>
                      <div className="grid grid-cols-2 gap-3 text-xs sm:text-sm">
                        {[
                          { label: "Version", value: "v2.1.0" },
                          { label: "Model", value: "CNN v2.1" },
                          { label: "Uptime", value: "99.9%" },
                          { label: "Status", value: "Online", color: "text-green-600" },
                        ].map((item) => (
                          <div key={item.label} className="text-center">
                            <div className="font-medium text-gray-900">{item.label}</div>
                            <div className={item.color || "text-gray-600"}>{item.value}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </TabsContent>
                </div>
              </ScrollArea>
            </div>

            {/* Mobile Navigation Footer */}
            <div className="sm:hidden border-t border-gray-200 p-4">
              <div className="flex justify-between items-center">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goToPrevTab}
                  disabled={currentTabIndex === 0}
                  className="flex items-center gap-2"
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goToNextTab}
                  disabled={currentTabIndex === tabs.length - 1}
                  className="flex items-center gap-2"
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  )
}
