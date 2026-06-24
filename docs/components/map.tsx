'use client'
import DottedMap from 'dotted-map'
import { useEffect, useState } from 'react'

const pins = [
	{ lat: 40.73061, lng: -73.935242 },
	{ lat: 48.8534, lng: 2.3488 },
	{ lat: 51.5074, lng: -0.1278 },
	{ lat: 35.6895, lng: 139.6917 },
	{ lat: 34.0522, lng: -118.2437 },
	{ lat: 55.7558, lng: 37.6173 },
	{ lat: 39.9042, lng: 116.4074 },
	{ lat: 19.4326, lng: -99.1332 },
	{ lat: 37.7749, lng: -122.4194 },
	{ lat: -33.8688, lng: 151.2093 },
	{ lat: 28.6139, lng: 77.209 },
	{ lat: 52.52, lng: 13.405 },
	{ lat: 41.9028, lng: 12.4964 },
	{ lat: 43.65107, lng: -79.347015 },
	{ lat: -23.55052, lng: -46.633308 },
	{ lat: 31.2304, lng: 121.4737 },
	{ lat: 55.9533, lng: -3.1883 },
	{ lat: 35.6762, lng: 139.6503 },
	{ lat: 1.3521, lng: 103.8198 },
	{ lat: 37.5665, lng: 126.978 },
	{ lat: 53.3498, lng: -6.2603 },
	{ lat: 30.0444, lng: 31.2357 },
	{ lat: 50.4501, lng: 30.5234 },
	{ lat: -34.6037, lng: -58.3816 },
	{ lat: 59.9343, lng: 30.3351 },
	{ lat: 25.276987, lng: 55.296249 },
	{ lat: 45.4642, lng: 9.19 },
	{ lat: -22.9068, lng: -43.1729 },
	{ lat: 40.4168, lng: -3.7038 },
	{ lat: 41.3851, lng: 2.1734 },
	{ lat: 13.7563, lng: 100.5018 },
	{ lat: 52.3676, lng: 4.9041 },
	{ lat: -37.8136, lng: 144.9631 },
	{ lat: 60.1695, lng: 24.9354 },
	{ lat: 47.4979, lng: 19.0402 },
	{ lat: 59.3293, lng: 18.0686 },
	{ lat: 35.9078, lng: 127.7669 },
	{ lat: 46.2044, lng: 6.1432 },
	{ lat: 29.7604, lng: -95.3698 },
	{ lat: 39.7392, lng: -104.9903 },
	{ lat: -11.6647, lng: 27.4794 },
	{ lat: -10.7026, lng: 25.5122 },
	{ lat: -4.4419, lng: 15.2663 },
]

function buildSvg(isDark: boolean) {
	const map = new DottedMap({ height: 55, grid: 'diagonal' })

	pins.forEach((pin) => {
		map.addPin({
			...pin,
			svgOptions: {
				color: isDark
					? '#a78bfa'  // bright purple on #050505
					: '#7c3aed', // vivid purple on #FAFAFA
				radius: 0.4,
			},
		})
	})

	return map.getSVG({
		radius: 0.22,
		color: isDark
			? '#555555'   // medium gray on #050505
			: '#a0a0a0',  // medium gray on #FAFAFA
		shape: 'circle',
		backgroundColor: 'transparent',
	})
}

export const Map = () => {
	const [isDark, setIsDark] = useState(false)

	useEffect(() => {
		const check = () => setIsDark(document.documentElement.classList.contains('dark'))
		check()
		const observer = new MutationObserver(check)
		observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
		return () => observer.disconnect()
	}, [])

	const svgMap = buildSvg(isDark)

	return (
		<img
			src={`data:image/svg+xml;utf8,${encodeURIComponent(svgMap)}`}
			alt="map illustration"
		/>
	)
}
