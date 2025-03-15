export default function ArtGallery() {
  return (
    <div className="container py-12">
      <h1 className="text-3xl font-bold mb-8">Art Gallery</h1>
      
      <p className="text-lg mb-8">
        Welcome to my art gallery. Here you can find a collection of my artwork and photos of me with my art.
      </p>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {/* Using direct HTML image tags with fixed dimensions */}
        <div className="art-card">
          <img src="/art/20221003_170829.jpg" alt="Artwork 1" width="400" height="300" 
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 1</p>
        </div>
        
        <div className="art-card">
          <img src="/art/20221003_171003.jpg" alt="Artwork 2" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 2</p>
        </div>
        
        <div className="art-card">
          <img src="/art/20221013_170405.jpg" alt="Artwork 3" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 3</p>
        </div>
        
        <div className="art-card">
          <img src="/art/20221013_174915.jpg" alt="Artwork 4" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 4</p>
        </div>
        
        <div className="art-card">
          <img src="/art/AfterlightImage1.jpg" alt="Artwork 5" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 5</p>
        </div>
        
        <div className="art-card">
          <img src="/art/C62D2136-EC80-4938-8033-C68628FC570A.jpg" alt="Artwork 6" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 6</p>
        </div>
        
        <div className="art-card">
          <img src="/art/C8C6DEF8-4239-4B16-ADF3-4EAF62D4795A.jpg" alt="Artwork 7" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 7</p>
        </div>
        
        <div className="art-card">
          <img src="/art/DA2C3455-82BB-4293-B62A-44D470647DFE.jpg" alt="Artwork 8" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 8</p>
        </div>
        
        <div className="art-card">
          <img src="/art/E1AFB3A0-D6AE-4705-83C9-3F440C307484.jpg" alt="Artwork 9" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 9</p>
        </div>

        <div className="art-card">
          <img src="/art/EBC7DCC4-332E-48AD-A2D1-8561AA1104F2.jpg" alt="Artwork 10" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 10</p>
        </div>

        <div className="art-card">
          <img src="/art/FB_IMG_1686089299754.jpg" alt="Artwork 11" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 11</p>
        </div>

        <div className="art-card">
          <img src="/art/FB_IMG_1686089748977.jpg" alt="Artwork 12" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 12</p>
        </div>
        
        <div className="art-card">
          <img src="/art/FB_IMG_1686089761548.jpg" alt="Artwork 13" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 13</p>
        </div>
        
        <div className="art-card">
          <img src="/art/FB_IMG_1686089772880.jpg" alt="Artwork 14" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 14</p>
        </div>
        
        <div className="art-card">
          <img src="/art/FB_IMG_1686089783737.jpg" alt="Artwork 15" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 15</p>
        </div>
        
        <div className="art-card">
          <img src="/art/FB_IMG_1686089791426.jpg" alt="Artwork 16" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 16</p>
        </div>
        
        <div className="art-card">
          <img src="/art/FB_IMG_1686089798139.jpg" alt="Artwork 17" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 17</p>
        </div>
        
        <div className="art-card">
          <img src="/art/FB_IMG_1686089803343.jpg" alt="Artwork 18" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 18</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0004.JPG" alt="Artwork 19" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 19</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0009.PNG" alt="Artwork 20" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 20</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0010.JPG" alt="Artwork 21" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 21</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0019.JPG" alt="Artwork 22" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 22</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0041.JPG" alt="Artwork 23" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 23</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0052.JPG" alt="Artwork 24" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 24</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0080.JPG" alt="Artwork 25" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 25</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0092.JPG" alt="Artwork 26" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 26</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0097.JPG" alt="Artwork 27" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 27</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0115.JPG" alt="Artwork 28" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 28</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0120.JPG" alt="Artwork 29" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 29</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0127.JPG" alt="Artwork 30" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 30</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0129.JPG" alt="Artwork 31" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 31</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0133.JPG" alt="Artwork 32" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 32</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0144.JPG" alt="Artwork 33" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 33</p>
        </div>
        
        <div className="art-card">
          <img src="/art/IMG_0149.JPG" alt="Artwork 34" width="400" height="300"
               className="object-cover rounded-lg shadow-md hover:shadow-xl transition-all duration-300" />
          <p className="mt-2 text-center text-sm text-gray-600">Artwork 34</p>
        </div>
      </div>
    </div>
  );
}
