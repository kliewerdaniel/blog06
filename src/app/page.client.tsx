'use client';

import Link from "next/link";
import Image from "next/image";
import PageTransition from "@/components/PageTransition";
import AnimatedBackground from "@/components/AnimatedBackground";
import { motion as m } from "framer-motion";
import { PostMetadata } from "@/types/blog";

interface ClientHomePageProps {
  recentPosts: PostMetadata[];
  projects: {
    title: string;
    description: string;
    technologies: string[];
    link: string;
  }[];
  technologies: {
    name: string;
    count: number;
  }[];
}

export default function ClientHomePage({ recentPosts, projects, technologies }: ClientHomePageProps) {
  return (
    <PageTransition transitionType="scale">
      {/* Background is now provided by the root layout */}
      
      <div className="min-h-screen">
        {/* Split screen hero layout */}
        <section className="grid grid-cols-1 lg:grid-cols-5 min-h-[60vh] w-full">
          {/* Left side: Image and name */}
          <div className="lg:col-span-2 flex flex-col justify-center items-center p-8 relative">
            <m.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="relative w-64 h-64 rounded-2xl overflow-hidden border-4 border-primary/20 drop-shadow-xl mb-8"
            >
              <Image 
                src="/self.jpg" 
                alt="Daniel Kliewer - AI Artist & Creative Technologist"
                fill
                style={{ objectFit: 'cover' }}
                priority
                className="hover:scale-105 transition-transform duration-500"
              />
            </m.div>
            <m.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-4xl md:text-5xl font-bold text-center"
            >
              Daniel Kliewer
            </m.h1>
            <m.h2 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="text-2xl text-primary text-center"
            >
              AI Artist & Creative Technologist
            </m.h2>
          </div>
          
          {/* Right side: Bio and CTA */}
          <div className="lg:col-span-3 flex flex-col justify-center p-8 lg:p-12">
            <m.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-secondary/10 backdrop-blur-sm p-8 rounded-2xl mb-8 border border-secondary/20 shadow-lg"
            >

              <p className="text-md mb-4">
                I explore the intersection of art and artificial intelligence, 
                using AI as both medium and collaborator. My work challenges traditional notions 
                of creativity while exploring the evolving relationship between humans and machines.
              </p>
              <p className="text-md">
                Through my work, I investigate themes of digital identity, algorithmic creativity, and 
                the blurred boundaries between human and machine artistic expression.
              </p>
            </m.div>
            
            <m.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="flex gap-4 flex-wrap justify-center"
            >
              <Link 
                href="/blog" 
                className="btn btn-primary text-lg px-8 py-3 shadow-md">
                Read My Blog
              </Link>
              <Link 
                href="/art" 
                className="btn btn-outline-primary text-lg px-8 py-3 shadow-md">
                View My Art
              </Link>
            </m.div>
          </div>
        </section>
        
        {/* Technology tags cloud */}
        <section className="py-12 container mx-auto px-4">
          <m.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="flex flex-wrap justify-center gap-3 mb-12"
          >
            {technologies.map((tech, index) => (
              <span 
                key={tech.name} 
                className="inline-block px-4 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-full text-sm md:text-base transition-colors duration-300 cursor-default"
                style={{
                  fontSize: `${Math.min(1.2 + (tech.count * 0.1), 1.8)}rem`,
                  opacity: 0.7 + (tech.count * 0.05)
                }}
              >
                {tech.name}
              </span>
            ))}
          </m.div>
        </section>
        
        {/* Featured Projects Showcase */}
        <section className="py-12 bg-gradient-to-b from-transparent to-muted/30 backdrop-blur-sm">
          <div className="container mx-auto px-4">
            <m.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.7 }}
              className="mb-12"
            >
              <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">Featured Projects</h2>
              <p className="text-muted-foreground text-center max-w-2xl mx-auto">
                Explore my latest projects combining AI, art, and technology to create innovative experiences
              </p>
            </m.div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {projects.map((project, index) => (
                <m.div
                  key={project.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.8 + (index * 0.1) }}
                  className="card p-6 transition-all hover:shadow-lg border-2 hover:-translate-y-1"
                >
                  <Link href={project.link}>
                    <h3 className="text-xl font-bold mb-3">{project.title}</h3>
                    <div className="mb-4 flex flex-wrap gap-2">
                      {project.technologies.map((tech, idx) => (
                        <span 
                          key={idx} 
                          className="inline-block text-xs px-2 py-1 bg-secondary/10 text-secondary rounded-full"
                        >
                          {tech}
                        </span>
                      ))}
                    </div>
                    <p className="text-muted-foreground mb-4">
                      {project.description}
                    </p>
                    <span className="text-primary hover:underline inline-flex items-center">
                      Learn more
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </span>
                  </Link>
                </m.div>
              ))}
            </div>
          </div>
        </section>
        
        {/* Recent Blog Posts Section */}
        <section className="py-16 container mx-auto px-4">
          <m.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 1.1 }}
            className="flex justify-between items-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold">Recent Blog Posts</h2>
            <Link href="/blog" className="text-primary hover:underline group flex items-center">
              View All 
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-1 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </Link>
          </m.div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {recentPosts.map((post, index) => (
              <m.div
                key={post.slug}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1.2 + (index * 0.1) }}
              >
                <Link 
                  href={`/blog/${post.slug}`}
                  className="card p-6 transition-all hover:shadow-md flex flex-col h-full hover:-translate-y-1"
                >
                  <h3 className="text-xl font-bold mb-3 line-clamp-2">{post.title}</h3>
                  
                  {post.categories && post.categories.length > 0 && (
                    <div className="mb-3 flex flex-wrap gap-2">
                      {post.categories.slice(0, 3).map((category: string, idx: number) => (
                        <span 
                          key={idx} 
                          className="inline-block text-xs px-2 py-1 bg-primary/10 text-primary rounded-full"
                        >
                          {category}
                        </span>
                      ))}
                      {post.categories.length > 3 && (
                        <span className="inline-block text-xs px-2 py-1 bg-muted text-muted-foreground rounded-full">
                          +{post.categories.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                  
                  <p className="text-sm text-muted-foreground mb-3">
                    {new Date(post.date).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </p>
                  
                  {post.excerpt && (
                    <p className="text-sm text-muted-foreground line-clamp-3 mb-4 flex-grow">
                      {post.excerpt}
                    </p>
                  )}
                  
                  <span className="text-primary hover:underline inline-flex items-center mt-auto">
                    Read more
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </span>
                </Link>
              </m.div>
            ))}
          </div>
        </section>

        {/* Final CTA Section */}
        <section className="py-16 bg-gradient-to-t from-transparent to-secondary/10 backdrop-blur-sm">
          <div className="container mx-auto px-4 text-center">
            <m.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1.6 }}
              className="max-w-2xl mx-auto"
            >
              <h2 className="text-3xl md:text-4xl font-bold mb-6">Let's Build Something Amazing</h2>
              <p className="text-lg mb-8">
                I'm always open to new projects, collaborations, and conversations about AI art and creative technology.
              </p>
              <Link 
                href="/blog" 
                className="btn btn-primary text-lg px-8 py-3 shadow-md hover:scale-105 transition-transform"
              >
                Explore My Work
              </Link>
            </m.div>
          </div>
        </section>

        {/* Shopify Product Section */}
        <section className="py-16 container mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-8">My Latest AI Art Collection</h2>
          <div className="max-w-4xl mx-auto">
            <div id='product-component-1744048152739'></div>
          </div>
          
          <script type="text/javascript" dangerouslySetInnerHTML={{
            __html: `
              (function () {
                var scriptURL = 'https://sdks.shopifycdn.com/buy-button/latest/buy-button-storefront.min.js';
                if (window.ShopifyBuy) {
                  if (window.ShopifyBuy.UI) {
                    ShopifyBuyInit();
                  } else {
                    loadScript();
                  }
                } else {
                  loadScript();
                }
                function loadScript() {
                  var script = document.createElement('script');
                  script.async = true;
                  script.src = scriptURL;
                  (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(script);
                  script.onload = ShopifyBuyInit;
                }
                function ShopifyBuyInit() {
                  var client = ShopifyBuy.buildClient({
                    domain: 'sjxbb1-wy.myshopify.com',
                    storefrontAccessToken: '9ca69e8e6153ed601e25456fede3f255',
                  });
                  ShopifyBuy.UI.onReady(client).then(function (ui) {
                    ui.createComponent('product', {
                      id: '7763998867508',
                      node: document.getElementById('product-component-1744048152739'),
                      moneyFormat: '%24%7B%7Bamount%7D%7D',
                      options: {
                        "product": {
                          "styles": {
                            "product": {
                              "@media (min-width: 601px)": {
                                "max-width": "calc(25% - 20px)",
                                "margin-left": "20px",
                                "margin-bottom": "50px"
                              }
                            }
                          },
                          "text": {
                            "button": "Add to cart"
                          }
                        },
                        "productSet": {
                          "styles": {
                            "products": {
                              "@media (min-width: 601px)": {
                                "margin-left": "-20px"
                              }
                            }
                          }
                        },
                        "modalProduct": {
                          "contents": {
                            "img": false,
                            "imgWithCarousel": true,
                            "button": false,
                            "buttonWithQuantity": true
                          },
                          "styles": {
                            "product": {
                              "@media (min-width: 601px)": {
                                "max-width": "100%",
                                "margin-left": "0px",
                                "margin-bottom": "0px"
                              }
                            }
                          },
                          "text": {
                            "button": "Add to cart"
                          }
                        },
                        "option": {},
                        "cart": {
                          "text": {
                            "total": "Subtotal",
                            "button": "Checkout"
                          }
                        },
                        "toggle": {}
                      }
                    });
                  });
                }
              })();
            `
          }} />
        </section>
      </div>
    </PageTransition>
  );
}
