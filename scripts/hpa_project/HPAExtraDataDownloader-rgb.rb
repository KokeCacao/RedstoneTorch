require 'net/http'

labels_dict={
  "Nucleoplasm"=>0,
  "Nuclear membrane"=>1,
  "Nucleoli"=>2,
  "Nucleoli fibrillar center"=>3,
  "Nuclear speckles"=>4,
  "Nuclear bodies"=>5,
  "Endoplasmic reticulum"=>6,
  "Golgi apparatus"=>7,
  "Peroxisomes"=>8,
  "Endosomes"=>9,
  "Lysosomes"=>10,
  "Intermediate filaments"=>11,
  "Actin filaments"=>12,
  "Focal adhesion sites"=>13,
  "Microtubules"=>14,
  "Microtubule ends"=>15,
  "Cytokinetic bridge"=>16,
  "Mitotic spindle"=>17,
  "Microtubule organizing center"=>18,
  "Centrosome"=>19,
  "Lipid droplets"=>20,
  "Plasma membrane"=>21,
  "Cell junctions"=>22,
  "Mitochondria"=>23,
  "Aggresome"=>24,
  "Cytosol"=>25,
  "Cytoplasmic bodies"=>26,
  "Rods & rings"=>27
}
labels_lower = {}

labels_dict.each do |k,v|
	labels_lower[k.downcase]=v
end


class HttpNoTcpDelay < Net::HTTP
  def on_connect
    @socket.io.setsockopt(Socket::IPPROTO_TCP, Socket::TCP_NODELAY, 1)
    nil
  end
end

in_data = false
locations = []
urls = []
entries = []
last_verification = "approved"
line_count = 0
ARGF.each_line do |line|
	line_count = line_count+1
	if line.include? "</data>"
		if locations.length > 0
			#puts "Added #{urls.length} images"
			#puts "LOCATIONS:"
			#puts "  #{locations.join(" ")}"
			#puts "URLS:"
			#urls.each{|url| puts "  #{url}"}
			entries << {
				:locations => locations.join(" "),
				:urls => urls
			}
		end
		in_data = false
		have_location = false
		locations = []
		urls = []
		next
	end
	if line.include? "</verification>" and line.include? 'type="validation"'
		last_verification = line.match(/>(.*)</)[1]
		puts "#{line_count}:current verification: #{last_verification}"

	end
	if not in_data and line.include? "<data>"
		in_data = true
		next
	end
	if in_data and last_verification != 'uncertain'
		re = />(.*)</
		if line.include? "<location"
			loc = line.match(re)[1]
			if labels_lower[loc]
				locations << labels_lower[loc]
			end
		elsif line.include? "_red" and locations.length > 0
			urls << line.match(re)[1]
		end
	end
end

puts "#{entries.length} entries to process"
image_id=0
out = open("augment.csv","w")
num_urls=0
urls=[]
entries.each{|e| num_urls += e[:urls].length }
entries.each do |entry|
	if entry[:urls].length > 0
		entry[:urls].each do |url|
			l = entry[:locations]
			out.write("#{image_id},#{l}\n")
			urls << { :id => image_id, :url => url }
			image_id = image_id + 1
		end
	end
end
out.close()
puts "collected #{urls.length} urls"
require 'thread'
mutex = Mutex.new
num_threads = 12
threads =  []
(0..num_threads).each do |i|
	thr = Thread.new do
		while true do
			img = nil
			mutex.synchronize do
				img = urls.pop
			end
			break if img.nil?
			image_id = img[:id]
			url = img[:url]
			f = "#{image_id}.jpg"
			fn= "images/#{image_id}_rgb.jpg"

			url.gsub!('http://','')
			d,p = url.split('.org/')
			d = "#{d}.org"
			if File.exist?(fn)
				puts("#{i}:skip existing #{fn}")
				next
 			else	
				puts "#{i}:#{urls.length}::#{d}/#{p}::#{f}"
			end
			begin
			        net = HttpNoTcpDelay.new(d,80)
			        net.start do |http|
			            resp = http.get("/#{p}")
			            open(fn, "wb") do |file|
			            #open(f, "wb") do |file|
			                file.write(resp.body)
			            end
			        end
			rescue
				puts "*RETRY"
				sleep(2)
				retry
			end
			#system("convert -resize 2048x2048 -define png:compression-filter=2 -define png:compression-level=9 -define png:compression-strategy=1 #{f} #{fn}")
			#system("rm #{f}")
		end
	end
	threads << thr
end

threads.each do |thr|
	thr.join
end
