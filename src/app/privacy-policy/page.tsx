import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "Privacy policy for Daniel Kliewer's website, detailing how we collect, use, and protect your data.",
};

export default function PrivacyPolicy() {
  return (
    <div className="container max-w-3xl py-12">
      <h1 className="text-3xl font-bold mb-8">Privacy Policy</h1>
      
      <div className="prose prose-invert max-w-none">
        <p className="text-lg mb-6">
          Last updated: March 10, 2025
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Introduction</h2>
        <p>
          This Privacy Policy describes how your personal information is collected, used, and shared when you visit danielkliewer.com (the "Site").
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Personal Information We Collect</h2>
        <p>
          When you visit the Site, we automatically collect certain information about your device, including information about your web browser, IP address, time zone, and some of the cookies that are installed on your device. Additionally, as you browse the Site, we collect information about the individual web pages that you view, what websites or search terms referred you to the Site, and information about how you interact with the Site. We refer to this automatically-collected information as "Device Information."
        </p>

        <h3 className="text-lg font-semibold mt-6 mb-3">We collect Device Information using the following technologies:</h3>
        <ul className="list-disc pl-6 mb-6">
          <li className="mb-2">
            <strong>Cookies:</strong> Data files that are placed on your device and often include an anonymous unique identifier. You can instruct your browser to refuse all cookies or to indicate when a cookie is being sent. However, if you do not accept cookies, you may not be able to use some portions of our Site.
          </li>
          <li className="mb-2">
            <strong>Log Files:</strong> Track actions occurring on the Site, and collect data including your IP address, browser type, Internet service provider, referring/exit pages, and date/time stamps.
          </li>
          <li className="mb-2">
            <strong>Google Analytics:</strong> Web analytics service that tracks and reports website traffic. All data is collected anonymously and without personal identifiers when you grant consent. If you choose to "Decline" analytics in the consent banner, or if your browser has "Do Not Track" enabled, no data will be collected by Google Analytics.
          </li>
        </ul>

        <h2 className="text-xl font-semibold mt-8 mb-4">How We Use Your Information</h2>
        <p>
          If you've accepted analytics tracking, we use the Device Information that we collect to help us screen for potential risk and fraud (in particular, your IP address), and more generally to improve and optimize our Site (for example, by generating analytics about how our customers browse and interact with the Site, and to assess the success of our marketing and advertising campaigns).
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Our Privacy-First Approach</h2>
        <p>
          We take a privacy-first approach to analytics with the following measures:
        </p>
        <ul className="list-disc pl-6 mb-6">
          <li className="mb-2">
            <strong>Explicit Consent:</strong> Analytics are only enabled after you explicitly consent via the cookie banner.
          </li>
          <li className="mb-2">
            <strong>Do Not Track Respected:</strong> If your browser has "Do Not Track" enabled, we automatically disable all analytics tracking regardless of consent status.
          </li>
          <li className="mb-2">
            <strong>IP Anonymization:</strong> All IP addresses are anonymized before being stored.
          </li>
          <li className="mb-2">
            <strong>No Cross-Site Tracking:</strong> We disable Google signals, ad personalization, and restrict data processing to further protect your privacy.
          </li>
          <li className="mb-2">
            <strong>No Personal Data:</strong> We never collect or store personally identifiable information through our analytics.
          </li>
          <li className="mb-2">
            <strong>Withdrawal of Consent:</strong> You can withdraw consent at any time by clearing your cookies for this site.
          </li>
        </ul>

        <h2 className="text-xl font-semibold mt-8 mb-4">Data Retention</h2>
        <p>
          If you've consented to analytics, anonymized usage data is stored for a maximum of 14 months in Google Analytics, after which it is automatically deleted.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Changes</h2>
        <p>
          We may update this privacy policy from time to time in order to reflect, for example, changes to our practices or for other operational, legal or regulatory reasons.
        </p>

        <h2 className="text-xl font-semibold mt-8 mb-4">Contact Us</h2>
        <p>
          For more information about our privacy practices, if you have questions, or if you would like to make a complaint, please contact us by e-mail at danielkliewer@gmail.com.
        </p>
      </div>
    </div>
  );
}
