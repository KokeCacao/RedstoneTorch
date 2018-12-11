require 'net/http'

text=File.open('subcellular_location.tsv').read
text.gsub!(/\r\n?/, "\n")
text.each_line do |line|
    split_line = line.split("\t")
    puts "LINE:#{split_line[0].to_s.strip}||#{split_line[3].to_s.strip}|#{split_line[4].to_s.strip}|#{split_line[5].to_s.strip}"
    Net::HTTP.start("www.proteinatlas.org") do |http|
        resp = http.get("/#{split_line[0].to_s.strip}.xml")
        open("xml/#{split_line[0].to_s.strip}.xml", "wb") do |file|
            file.write(resp.body)
        end
    end
end
