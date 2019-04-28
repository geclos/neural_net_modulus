file = File.new("data.csv",  "w+")

(0..10000).each do |_|
  number = ( '%08d' % (Random.rand(99999999).to_i) )
  letter = 'TRWAGMYFPDXBNJZSQVHLCKE'[number.to_i % 23]

  file << "#{number},#{letter}\n"
  putc '.'
end

file.close
