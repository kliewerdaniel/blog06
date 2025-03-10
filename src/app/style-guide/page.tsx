import Link from "next/link";

export const metadata = {
  title: "Style Guide | Design System",
  description: "A comprehensive guide to our design system including colors, typography, and UI components",
};

export default function StyleGuidePage() {
  return (
    <div className="container py-12">
      <h1>Design System Style Guide</h1>
      <p className="text-lg mb-8">
        This page showcases our design system with standardized colors, typography, and UI components.
      </p>

      {/* Color Palette Section */}
      <section className="mb-12">
        <h2 id="colors">Color Palette</h2>
        <p className="mb-6">Our color system consists of primary, secondary, and accent colors with consistent naming and application.</p>
        
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">Primary Colors (Blue)</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            <ColorSwatch name="50" color="bg-primary-50" textColor="text-foreground" />
            <ColorSwatch name="100" color="bg-primary-100" textColor="text-foreground" />
            <ColorSwatch name="300" color="bg-primary-300" textColor="text-foreground" />
            <ColorSwatch name="500 (Primary)" color="bg-primary" textColor="text-white" />
            <ColorSwatch name="700" color="bg-primary-700" textColor="text-white" />
          </div>
        </div>
        
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">Secondary Colors (Purple)</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            <ColorSwatch name="50" color="bg-secondary-50" textColor="text-foreground" />
            <ColorSwatch name="100" color="bg-secondary-100" textColor="text-foreground" />
            <ColorSwatch name="300" color="bg-secondary-300" textColor="text-foreground" />
            <ColorSwatch name="500 (Secondary)" color="bg-secondary" textColor="text-white" />
            <ColorSwatch name="700" color="bg-secondary-700" textColor="text-white" />
          </div>
        </div>
        
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">Accent Colors (Teal)</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            <ColorSwatch name="50" color="bg-accent-50" textColor="text-foreground" />
            <ColorSwatch name="100" color="bg-accent-100" textColor="text-foreground" />
            <ColorSwatch name="300" color="bg-accent-300" textColor="text-foreground" />
            <ColorSwatch name="500 (Accent)" color="bg-accent" textColor="text-white" />
            <ColorSwatch name="700" color="bg-accent-700" textColor="text-white" />
          </div>
        </div>
        
        <div>
          <h3 className="text-lg font-semibold mb-4">Neutral & System Colors</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
            <ColorSwatch name="Background" color="bg-background" textColor="text-foreground" border />
            <ColorSwatch name="Foreground" color="bg-foreground" textColor="text-white" />
            <ColorSwatch name="Muted" color="bg-muted" textColor="text-foreground" />
            <ColorSwatch name="Border" color="bg-border" textColor="text-foreground" />
            <ColorSwatch name="Success" color="bg-success" textColor="text-white" />
            <ColorSwatch name="Warning" color="bg-warning" textColor="text-white" />
            <ColorSwatch name="Danger" color="bg-danger" textColor="text-white" />
            <ColorSwatch name="Info" color="bg-info" textColor="text-white" />
          </div>
        </div>
      </section>

      {/* Typography Section */}
      <section className="mb-12">
        <h2 id="typography">Typography</h2>
        <p className="mb-6">Our typography system uses Geist Sans as the primary font with a clear hierarchy.</p>
        
        <div className="space-y-6 mb-8">
          <div>
            <h1>Heading 1 (2.25rem/36px)</h1>
            <p className="text-muted-foreground">Used for main page titles</p>
          </div>
          
          <div>
            <h2>Heading 2 (1.875rem/30px)</h2>
            <p className="text-muted-foreground">Used for section headings</p>
          </div>
          
          <div>
            <h3>Heading 3 (1.5rem/24px)</h3>
            <p className="text-muted-foreground">Used for subsections</p>
          </div>
          
          <div>
            <h4>Heading 4 (1.25rem/20px)</h4>
            <p className="text-muted-foreground">Used for card titles</p>
          </div>
          
          <div>
            <h5>Heading 5 (1.125rem/18px)</h5>
            <p className="text-muted-foreground">Used for smaller section titles</p>
          </div>
          
          <div>
            <h6>Heading 6 (1rem/16px)</h6>
            <p className="text-muted-foreground">Used for minor headings</p>
          </div>
        </div>
        
        <div className="space-y-6">
          <div>
            <p className="text-lg">Large Paragraph (1.125rem/18px)</p>
            <p className="text-lg">Used for important text blocks, introductions, or featured content. The spacing and line height provide excellent readability.</p>
          </div>
          
          <div>
            <p>Body Text (1rem/16px)</p>
            <p>The standard body text is used for the main content. It uses a line height of 1.6 for optimal readability and has a paragraph spacing of 1.5em.</p>
            <p>This shows the spacing between multiple paragraphs. Note how the consistent spacing makes the text easy to read and scan.</p>
          </div>
          
          <div>
            <p className="text-sm">Small Text (0.875rem/14px)</p>
            <p className="text-sm">Used for supporting text, captions, metadata, or less important information that doesn't need to be emphasized.</p>
          </div>
        </div>
      </section>

      {/* Buttons Section */}
      <section className="mb-12">
        <h2 id="buttons">Buttons</h2>
        <p className="mb-6">Our button system provides consistent styling with clear visual hierarchy and states.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Button Variants</h3>
            <div className="flex flex-wrap gap-4">
              <button className="btn btn-primary">Primary Button</button>
              <button className="btn btn-secondary">Secondary Button</button>
              <button className="btn btn-accent">Accent Button</button>
              <button className="btn btn-outline">Outline Button</button>
              <button className="btn btn-ghost">Ghost Button</button>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Button Sizes</h3>
            <div className="flex flex-wrap items-center gap-4">
              <button className="btn btn-primary btn-sm">Small</button>
              <button className="btn btn-primary">Default</button>
              <button className="btn btn-primary btn-lg">Large</button>
            </div>
          </div>
        </div>
        
        <div>
          <h3 className="text-lg font-semibold mb-4">Button States</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <button className="btn btn-primary w-full">Default</button>
              <p className="text-sm text-muted-foreground">Normal state</p>
            </div>
            <div className="space-y-2">
              <button className="btn btn-primary w-full" disabled>Disabled</button>
              <p className="text-sm text-muted-foreground">Inactive state</p>
            </div>
            <div className="space-y-2">
              <button className="btn btn-primary w-full focus">Focus</button>
              <p className="text-sm text-muted-foreground">When focused via keyboard</p>
            </div>
          </div>
        </div>
      </section>

      {/* Forms Section */}
      <section className="mb-12">
        <h2 id="forms">Form Elements</h2>
        <p className="mb-6">Our form elements provide a consistent user experience with clear visual feedback.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="form-group">
              <label htmlFor="text-input">Text Input</label>
              <input 
                type="text" 
                id="text-input"
                placeholder="Enter your name"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="disabled-input">Disabled Input</label>
              <input 
                type="text" 
                id="disabled-input"
                placeholder="This field is disabled"
                disabled
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="select-input">Select Input</label>
              <select id="select-input">
                <option value="">Select an option</option>
                <option value="1">Option 1</option>
                <option value="2">Option 2</option>
                <option value="3">Option 3</option>
              </select>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="form-group">
              <label htmlFor="textarea">Textarea</label>
              <textarea 
                id="textarea"
                placeholder="Enter your message"
                rows={4}
              ></textarea>
            </div>
            
            <div className="form-group flex items-center gap-2">
              <input 
                type="checkbox" 
                id="checkbox"
              />
              <label htmlFor="checkbox" className="cursor-pointer">
                I agree to the terms and conditions
              </label>
            </div>
            
            <div className="form-group flex items-center gap-2">
              <input 
                type="radio" 
                id="radio1"
                name="radio-group"
              />
              <label htmlFor="radio1" className="cursor-pointer">
                Option A
              </label>
            </div>
            
            <div className="form-group flex items-center gap-2">
              <input 
                type="radio" 
                id="radio2"
                name="radio-group"
              />
              <label htmlFor="radio2" className="cursor-pointer">
                Option B
              </label>
            </div>
          </div>
        </div>
      </section>

      {/* Cards Section */}
      <section className="mb-12">
        <h2 id="cards">Cards</h2>
        <p className="mb-6">Cards are used to group related content and actions.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="card">
            <div className="card-header">
              <h4 className="m-0">Card Title</h4>
            </div>
            <div className="card-body">
              <p>This is a standard card with a header, body, and footer. Cards are used to group related content and actions.</p>
            </div>
            <div className="card-footer">
              <button className="btn btn-primary">Action</button>
            </div>
          </div>
          
          <div className="card">
            <div className="p-4">
              <h4 className="mb-2">Simple Card</h4>
              <p>A simpler card design without distinct header and footer sections. Useful for less complex content.</p>
              <div className="flex justify-end mt-4">
                <button className="btn btn-outline">Learn More</button>
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="card-body">
              <h4 className="mb-2">Feature Card</h4>
              <p>Feature cards highlight important information or functionality. They can be used to showcase key benefits or services.</p>
              <ul className="list-disc list-inside my-4">
                <li>Feature one description</li>
                <li>Feature two description</li>
                <li>Feature three description</li>
              </ul>
              <button className="btn btn-secondary w-full">Get Started</button>
            </div>
          </div>
        </div>
      </section>

      {/* Alerts Section */}
      <section className="mb-12">
        <h2 id="alerts">Alerts</h2>
        <p className="mb-6">Alerts are used to communicate important messages to users.</p>
        
        <div className="space-y-4">
          <div className="alert alert-info">
            <strong>Info:</strong> This is an informational alert message.
          </div>
          
          <div className="alert alert-success">
            <strong>Success:</strong> Your changes have been saved successfully.
          </div>
          
          <div className="alert alert-warning">
            <strong>Warning:</strong> Be careful, this action might have consequences.
          </div>
          
          <div className="alert alert-danger">
            <strong>Error:</strong> Something went wrong. Please try again.
          </div>
        </div>
      </section>

      {/* Spacing Section */}
      <section className="mb-12">
        <h2 id="spacing">Spacing System</h2>
        <p className="mb-6">Our design system uses an 8px spacing scale for consistency.</p>
        
        <div className="space-y-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Vertical Spacing</h3>
            <div className="bg-muted p-4 rounded">
              <div className="bg-primary py-2 text-center text-white mb-1">4px (0.25rem)</div>
              <div className="bg-primary py-2 text-center text-white mb-2">8px (0.5rem)</div>
              <div className="bg-primary py-2 text-center text-white mb-3">12px (0.75rem)</div>
              <div className="bg-primary py-2 text-center text-white mb-4">16px (1rem)</div>
              <div className="bg-primary py-2 text-center text-white mb-6">24px (1.5rem)</div>
              <div className="bg-primary py-2 text-center text-white mb-8">32px (2rem)</div>
              <div className="bg-primary py-2 text-center text-white">48px (3rem)</div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Horizontal Spacing</h3>
            <div className="bg-muted p-4 rounded flex items-center">
              <div className="bg-secondary h-16 w-1 text-white"></div>
              <div className="bg-secondary h-16 w-2 text-white ml-2"></div>
              <div className="bg-secondary h-16 w-3 text-white ml-3"></div>
              <div className="bg-secondary h-16 w-4 text-white ml-4"></div>
              <div className="bg-secondary h-16 w-6 text-white ml-6"></div>
              <div className="bg-secondary h-16 w-8 text-white ml-8"></div>
              <div className="bg-secondary h-16 w-12 text-white ml-12"></div>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Elements Section */}
      <section>
        <h2 id="interactive">Interactive Elements</h2>
        <p className="mb-6">These elements demonstrate our transition and animation styles.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Hover and Focus States</h3>
            <div className="space-y-4">
              <Link href="#" className="block p-4 bg-muted rounded transition-all hover:bg-primary hover:text-white">
                Hover over me to see the transition effect (0.2s)
              </Link>
              
              <button className="btn btn-primary w-full">
                Hover to see subtle transform effect
              </button>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4">Transitions</h3>
            <div className="space-y-4">
              <div className="p-4 bg-muted rounded transition-all hover:shadow-lg cursor-pointer">
                Hover to see shadow transition
              </div>
              
              <div className="relative p-4 bg-muted rounded overflow-hidden">
                <div className="transition-transform duration-300 hover:translate-x-2">
                  Hover to see transform transition →
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

// Color swatch component
function ColorSwatch({ name, color, textColor, border = false }: { name: string, color: string, textColor: string, border?: boolean }) {
  const classes = `${color} ${textColor} ${border ? 'border border-border' : ''} p-4 rounded flex flex-col items-center justify-center h-24`;
  
  return (
    <div>
      <div className={classes}>
        <span>{name}</span>
      </div>
    </div>
  );
}
